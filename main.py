import gradio as gr
import os
import glob
import csv
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import colorsys
import time


def ensure_dirs():
    Path("runs/annotated").mkdir(parents=True, exist_ok=True)
    Path("runs/logs").mkdir(parents=True, exist_ok=True)


def seeded_color_for_id(track_id: int) -> tuple:
    """Deterministic, visually distinct BGR color from an integer ID."""
    h = (int(track_id) * 0.61803398875) % 1.0
    s, v = 0.75, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_contours_from_masks(frame, masks, color, thickness=2):
    """Draws contour lines for each instance mask on the frame."""
    if masks is None:
        return
    
    if isinstance(masks, (list, tuple)):
        mask_list = masks
    else:
        mask_list = [masks] if masks.ndim == 2 else [m for m in masks]
    
    for m in mask_list:
        mm = (m.astype(np.uint8) * 255) if m.dtype != np.uint8 else m
        if mm.max() == 1:
            mm = mm * 255
        contours, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.polylines(frame, contours, isClosed=True, color=color, thickness=thickness)


def process_video(video_file, model_choice="yolov8n-seg.pt", conf_threshold=0.25, progress=gr.Progress()):
    """Process uploaded video with tracking."""
    ensure_dirs()
    
    if video_file is None:
        return None, None, "Please upload a video file"
    
    progress(0, desc="Loading model...")
    
    # Load model
    model_path = model_choice
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"
    
    # Initialize tracker
    tracker = sv.ByteTrack()
    
    progress(0.1, desc="Opening video...")
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None, None, "Could not open video file"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    # Setup output - Use H264 codec for better browser compatibility
    stem = Path(video_file).stem
    timestamp = int(time.time())
    out_video_path = f"runs/annotated/{stem}_{timestamp}_tracked.mp4"
    out_log_path = f"runs/logs/{stem}_{timestamp}_tracks.csv"
    
    # Try H264 codec first (better compatibility), fallback to mp4v
    fourcc_options = [
        ('avc1', cv2.VideoWriter_fourcc(*"avc1")),  # H264
        ('H264', cv2.VideoWriter_fourcc(*"H264")),  # H264 alternative
        ('X264', cv2.VideoWriter_fourcc(*"X264")),  # H264 alternative
        ('mp4v', cv2.VideoWriter_fourcc(*"mp4v")),  # MPEG-4
    ]
    
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    # Process video
    writer = None
    for codec_name, fourcc in fourcc_options:
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using codec: {codec_name}")
            break
        writer.release()
    
    if writer is None or not writer.isOpened():
        cap.release()
        return None, None, "Failed to initialize video writer"
    
    # Process video
    tracking_data = []
    frame_idx = 0
    t0 = time.perf_counter()
    
    progress(0.2, desc="Processing frames...")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # Run detection
        results = model(frame, imgsz=640, conf=conf_threshold, verbose=False)
        r0 = results[0]
        
        # Convert to Supervision format
        detections = sv.Detections.from_ultralytics(r0)
        
        # Filter for persons only
        person_class_ids = []
        try:
            names = r0.names
            if isinstance(names, dict):
                inv = {v: int(k) for k, v in names.items()}
                if "person" in inv:
                    person_class_ids = [inv["person"]]
            elif isinstance(names, list) and "person" in names:
                person_class_ids = [names.index("person")]
        except Exception:
            pass
        
        if person_class_ids:
            mask = np.isin(detections.class_id, person_class_ids)
            detections = detections[mask]
        
        # Update tracker
        detections = tracker.update_with_detections(detections)
        
        # Draw contours and labels
        for i in range(len(detections)):
            tid = detections.tracker_id[i]
            if tid is None:
                continue
            
            color = seeded_color_for_id(tid)
            
            # Draw contour
            if detections.mask is not None:
                m = detections.mask[i]
                draw_contours_from_masks(frame, m, color=color, thickness=2)
            
            # Draw label
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            label = f"ID {int(tid)}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # Log data
            conf_i = float(detections.confidence[i]) if detections.confidence is not None else -1.0
            tracking_data.append({
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "track_id": int(tid),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": conf_i
            })
        
        # Add FPS overlay
        dt = time.perf_counter() - t0
        curr_fps = (frame_idx + 1) / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {curr_fps:.1f}", (12, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        writer.write(frame)
        frame_idx += 1
        
        # Update progress
        if frame_idx % 10 == 0 and total_frames > 0:
            progress((0.2 + 0.7 * (frame_idx / total_frames)), 
                    desc=f"Processing frame {frame_idx}/{total_frames}")
    
    cap.release()
    writer.release()
    
    # Save CSV
    progress(0.95, desc="Saving tracking data...")
    df = pd.DataFrame(tracking_data)
    df.to_csv(out_log_path, index=False)
    
    # Verify the output file exists and has content
    if not os.path.exists(out_video_path):
        return None, None, "Error: Output video file was not created"
    
    file_size = os.path.getsize(out_video_path)
    if file_size < 1024:  # Less than 1KB is probably an error
        return None, None, f"Error: Output video file is too small ({file_size} bytes)"
    
    progress(1.0, desc="Complete!")
    
    stats = f"""
    Processing complete!
    
    Statistics:
    - Total frames: {frame_idx}
    - Unique IDs tracked: {len(df['track_id'].unique()) if len(df) > 0 else 0}
    - Average processing FPS: {frame_idx / (time.perf_counter() - t0):.1f}
    - Video saved: {out_video_path}
    - Logs saved: {out_log_path}
    """
    
    return out_video_path, out_log_path, stats


def process_webcam(model_choice, conf_threshold, duration=10):
    """Process webcam feed (placeholder for real-time processing)."""
    return None, None, "Webcam processing coming soon! Use video upload for now."


def load_existing_videos():
    """Load list of already processed videos."""
    annotated_dir = Path("runs/annotated")
    if not annotated_dir.exists():
        return []
    
    videos = list(annotated_dir.glob("*.mp4"))
    return [(str(v), v.stem) for v in videos]


def load_tracking_data(video_path):
    """Load tracking CSV for selected video."""
    if not video_path:
        return None
    
    stem = Path(video_path).stem.replace("_tracked", "")
    csv_path = Path("runs/logs") / f"{stem}_tracks.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None


# Gradio Interface
with gr.Blocks(title="Video Tracking Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Multi-Object Video Tracking System
    Upload a video or use webcam to track people with persistent IDs and contour visualization.
    """)
    
    with gr.Tabs():
        # Tab 1: Process New Video
        with gr.Tab(" Process Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video", sources=["upload"])
                    process_btn = gr.Button(" Process Video", variant="primary")
                
                with gr.Column(scale=1):
                    output_video = gr.Video(label="Annotated Video", autoplay=True)
                    output_status = gr.Textbox(label="Status", lines=10)
                    download_csv = gr.File(label="Download Tracking Data (CSV)")
            
            process_btn.click(
                fn=process_video,
                inputs=[video_input],
                outputs=[output_video, download_csv, output_status]
            )
        
 
        # # Tab 2: Webcam (Future)
        # with gr.Tab("WebCam"):
        #     with gr.Row():
        #         with gr.Column(scale=1):
        #             webcam_input = gr.Video(
        #                 label="Record from Webcam",
        #                 sources=["webcam"],
        #                 include_audio=False,  # Disable audio for faster processing
        #                 mirror_webcam=True
        #             )
        #             process_webcam_btn = gr.Button(" Process Video", variant="primary")
                
        #         with gr.Column(scale=1):
        #             webcam_output_video = gr.Video(label="Annotated Video")
        #             webcam_output_status = gr.Textbox(label="Status", lines=10)
        #             webcam_download_csv = gr.File(label="Download Tracking Data (CSV)")
            
        #     process_webcam_btn.click(
        #         fn=process_video,
        #         inputs=[webcam_input],
        #         outputs=[webcam_output_video, webcam_download_csv, webcam_output_status]
        #     )


if __name__ == "__main__":
    ensure_dirs()
    demo.launch(share=False)