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
from collections import defaultdict, deque


def ensure_dirs():
    Path("runs/annotated").mkdir(parents=True, exist_ok=True)
    Path("runs/logs").mkdir(parents=True, exist_ok=True)
    Path("runs/snapshots").mkdir(parents=True, exist_ok=True)


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


def draw_trajectory(frame, trajectory, color, max_points=50):
    """Draw trajectory path for a tracked object."""
    if len(trajectory) < 2:
        return
    
    # Limit trajectory length for performance
    points = list(trajectory)[-max_points:]
    
    # Draw lines connecting trajectory points
    for i in range(len(points) - 1):
        if points[i] is not None and points[i + 1] is not None:
            # Fade effect: older points are more transparent
            alpha = (i + 1) / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, points[i], points[i + 1], color, thickness, cv2.LINE_AA)
    
    # Draw circles at trajectory points
    for i, point in enumerate(points):
        if point is not None:
            alpha = (i + 1) / len(points)
            radius = max(2, int(4 * alpha))
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)


def draw_summary_panel(frame, active_ids, confidences, entries, exits, fps):
    """Draw live summary panel on video."""
    panel_height = 180
    panel_width = 350
    padding = 15
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
    
    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 2
    line_height = 25
    
    y_pos = 35
    
    # Title
    cv2.putText(frame, "TRACKING SUMMARY", (padding + 5, y_pos), 
                font, 0.7, (0, 255, 255), thickness, cv2.LINE_AA)
    y_pos += line_height + 5
    
    # Active people count
    cv2.putText(frame, f"Active People: {len(active_ids)}", (padding + 5, y_pos), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    y_pos += line_height
    
    # Active IDs
    if active_ids:
        ids_text = f"IDs: {', '.join(map(str, sorted(active_ids)))}"
        if len(ids_text) > 40:
            ids_text = ids_text[:37] + "..."
        cv2.putText(frame, ids_text, (padding + 5, y_pos), 
                    font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        cv2.putText(frame, "IDs: None", (padding + 5, y_pos), 
                    font, font_scale, color, thickness, cv2.LINE_AA)
    y_pos += line_height
    
    # Average confidence
    avg_conf = np.mean(confidences) if confidences else 0.0
    cv2.putText(frame, f"Avg Confidence: {avg_conf:.2f}", (padding + 5, y_pos), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    y_pos += line_height
    
    # Entries/Exits
    cv2.putText(frame, f"Entries: {entries} | Exits: {exits}", (padding + 5, y_pos), 
                font, font_scale, (0, 255, 0) if entries > exits else (0, 165, 255), 
                thickness, cv2.LINE_AA)
    y_pos += line_height
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (padding + 5, y_pos), 
                font, font_scale, (0, 255, 0) if fps > 20 else (0, 165, 255), 
                thickness, cv2.LINE_AA)


def save_snapshot(frame, bbox, track_id, output_dir):
    """Save cropped image of detected person."""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
    
    # Add padding
    pad = 10
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    
    cropped = frame[y1:y2, x1:x2]
    
    if cropped.size > 0:
        snapshot_path = Path(output_dir) / f"person_id_{track_id}.jpg"
        cv2.imwrite(str(snapshot_path), cropped)
        return str(snapshot_path)
    return None


def process_video(video_file, model_choice="yolo11n-seg.pt", conf_threshold=0.25, 
                  enable_trajectories=True, enable_snapshots=True, max_trajectory_points=50,
                  detect_all_objects=False, progress=gr.Progress()):
    """Process uploaded video with tracking and enhanced features."""
    ensure_dirs()
    
    if video_file is None:
        return None, None, None, "Please upload a video file"
    
    progress(0, desc="Loading model...")
    
    # Load model
    model_path = model_choice
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        return None, None, None, f"Error loading model: {str(e)}"
    
    # Initialize tracker
    tracker = sv.ByteTrack()
    
    progress(0.1, desc="Opening video...")
    
    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None, None, None, "Could not open video file"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    stem = Path(video_file).stem
    timestamp = int(time.time())
    out_video_path = f"runs/annotated/{stem}_{timestamp}_tracked.mp4"
    out_log_path = f"runs/logs/{stem}_{timestamp}_tracks.csv"
    out_events_path = f"runs/logs/{stem}_{timestamp}_events.csv"
    snapshot_dir = f"runs/snapshots/{stem}_{timestamp}"
    
    if enable_snapshots:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    
    # Try H264 codec first (better compatibility), fallback to mp4v
    fourcc_options = [
        ('avc1', cv2.VideoWriter_fourcc(*"avc1")),
        ('H264', cv2.VideoWriter_fourcc(*"H264")),
        ('X264', cv2.VideoWriter_fourcc(*"X264")),
        ('mp4v', cv2.VideoWriter_fourcc(*"mp4v")),
    ]
    
    writer = None
    for codec_name, fourcc in fourcc_options:
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"Using codec: {codec_name}")
            break
        writer.release()
    
    if writer is None or not writer.isOpened():
        cap.release()
        return None, None, None, "Failed to initialize video writer"
    
    # Tracking state
    tracking_data = []
    events_data = []
    trajectories = defaultdict(lambda: deque(maxlen=max_trajectory_points))
    seen_ids = set()
    active_ids_previous = set()
    snapshot_saved = set()
    
    total_entries = 0
    total_exits = 0
    
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
        
        # Debug: Log total detections before filtering
        total_detections_before = len(detections)
        
        # Filter for persons only (unless detect_all_objects is enabled)
        if not detect_all_objects and len(detections) > 0 and detections.class_id is not None:
            # In COCO dataset, person is always class 0
            person_mask = detections.class_id == 0
            detections = detections[person_mask]
        
        # Debug logging on first frame
        # if frame_idx == 0:
        #     print(f"Frame 0 Debug Info:")
        #     print(f"  - Total detections: {total_detections_before}")
        #     print(f"  - After filtering: {len(detections)}")
        #     print(f"  - Detect all objects: {detect_all_objects}")
        #     print(f"  - Model classes: {r0.names}")
        #     print(f"  - Confidence threshold: {conf_threshold}")
        
        # Update tracker
        detections = tracker.update_with_detections(detections)
        
        # Current frame active IDs
        active_ids = set()
        current_confidences = []
        
        # Process detections
        for i in range(len(detections)):
            tid = detections.tracker_id[i]
            if tid is None:
                continue
            
            tid = int(tid)
            active_ids.add(tid)
            
            color = seeded_color_for_id(tid)
            
            # Draw contour
            if detections.mask is not None:
                m = detections.mask[i]
                draw_contours_from_masks(frame, m, color=color, thickness=2)
            
            # Get bbox
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            
            # Calculate center point for trajectory
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center_point = (center_x, center_y)
            
            # Update trajectory
            if enable_trajectories:
                trajectories[tid].append(center_point)
                draw_trajectory(frame, trajectories[tid], color, max_trajectory_points)
            
            # Draw label
            label = f"ID {tid}"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # Save snapshot on first detection
            if enable_snapshots and tid not in snapshot_saved:
                snapshot_path = save_snapshot(frame, (x1, y1, x2, y2), tid, snapshot_dir)
                if snapshot_path:
                    snapshot_saved.add(tid)
            
            # Log tracking data
            conf_i = float(detections.confidence[i]) if detections.confidence is not None else -1.0
            current_confidences.append(conf_i)
            
            tracking_data.append({
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "track_id": tid,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
                "confidence": conf_i
            })
        
        # Detect entries and exits
        new_entries = active_ids - active_ids_previous
        new_exits = active_ids_previous - active_ids
        
        for tid in new_entries:
            if tid not in seen_ids:
                seen_ids.add(tid)
                total_entries += 1
                events_data.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / fps,
                    "event": "entry",
                    "track_id": tid
                })
        
        for tid in new_exits:
            total_exits += 1
            events_data.append({
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "event": "exit",
                "track_id": tid
            })
        
        active_ids_previous = active_ids.copy()
        
        # Calculate current FPS
        dt = time.perf_counter() - t0
        curr_fps = (frame_idx + 1) / dt if dt > 0 else 0.0
        
        # Draw summary panel
        draw_summary_panel(frame, active_ids, current_confidences, 
                          total_entries, total_exits, curr_fps)
        
        writer.write(frame)
        frame_idx += 1
        
        # Update progress
        if frame_idx % 10 == 0 and total_frames > 0:
            progress((0.2 + 0.7 * (frame_idx / total_frames)), 
                    desc=f"Processing frame {frame_idx}/{total_frames}")
    
    cap.release()
    writer.release()
    
    # Save tracking CSV
    progress(0.95, desc="Saving tracking data...")
    df_tracks = pd.DataFrame(tracking_data)
    df_tracks.to_csv(out_log_path, index=False)
    
    # Save events CSV
    df_events = pd.DataFrame(events_data)
    df_events.to_csv(out_events_path, index=False)
    
    # Verify output
    if not os.path.exists(out_video_path):
        return None, None, None, "Error: Output video file was not created"
    
    file_size = os.path.getsize(out_video_path)
    if file_size < 1024:
        return None, None, None, f"Error: Output video file is too small ({file_size} bytes)"
    
    progress(1.0, desc="Complete!")
    
    # Create zip of snapshots if enabled
    snapshot_zip = None
    if enable_snapshots and len(snapshot_saved) > 0:
        import shutil
        snapshot_zip = f"{snapshot_dir}.zip"
        shutil.make_archive(snapshot_dir, 'zip', snapshot_dir)
    
    # Calculate detection statistics
    detection_rate = (len(df_tracks) / frame_idx * 100) if frame_idx > 0 else 0
    frames_with_detections = len(df_tracks['frame_idx'].unique()) if len(df_tracks) > 0 else 0
    
    stats = f"""
    Processing Complete!
    
    Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Total Frames: {frame_idx}
    • Frames with Detections: {frames_with_detections} ({frames_with_detections/frame_idx*100:.1f}%)
    • Total Detections: {len(df_tracks)}
    • Unique People Tracked: {len(seen_ids)}
    • Total Entries: {total_entries}
    • Total Exits: {total_exits}
    • Average FPS: {frame_idx / (time.perf_counter() - t0):.1f}
    
     Output Files:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Video: {out_video_path}
    • Tracking Data: {out_log_path}
    • Events Log: {out_events_path}
    • Snapshots: {len(snapshot_saved)} saved
    
    """
    
    return out_video_path, out_log_path, snapshot_zip, stats


# Gradio Interface
with gr.Blocks(title="Video Tracking Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Advanced Multi-Object Video Tracking System
    Track people with persistent IDs, trajectory visualization, entry/exit detection, and more!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label=" Upload Video", sources=["upload"])          
            process_btn = gr.Button(" Process Video", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_video = gr.Video(label=" Annotated Video", autoplay=True)
            output_status = gr.Textbox(label=" Processing Status", lines=12)
            
            with gr.Row():
                download_csv = gr.File(label=" Tracking Data (CSV)")
                download_snapshots = gr.File(label=" Person Snapshots (ZIP)")
    
    gr.Markdown("""
    ###  Features:
    - **Persistent ID Tracking**: Each person maintains the same ID throughout the video
    - **Trajectory Visualization**: See the path each person takes
    - **Entry/Exit Detection**: Automatically log when people enter or leave the scene
    - **Live Summary Panel**: Real-time statistics overlay on video
    - **Person Snapshots**: Automatic cropped images of each unique person
    - **Performance Metrics**: Live FPS display
    """)
    
    process_btn.click(
        fn=process_video,
        inputs=[
            video_input
        ],
        outputs=[output_video, download_csv, download_snapshots, output_status]
    )


if __name__ == "__main__":
    ensure_dirs()
    demo.launch(share=False)