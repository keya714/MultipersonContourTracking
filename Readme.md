# Multi-Person Contour Tracking Dashboard

## Project Overview
This project implements a **computer vision pipeline** for detecting, tracking, and visualizing multiple people in video feeds using open-source tools like **YOLO**, **Supervision**, and **ByteTrack**.

The system is designed to perform **multiple person tracking** with persistent IDs, contour visualization, and structured data logging — suitable for real-world applications such as surveillance, retail analytics, or crowd management.

---

## Core Features

### 1. Video Ingestion
- Accepts both **recorded video files**.

### 2. Entity Detection
- Uses **YOLO-based models** (`YOLO11n-seg` as it is the fastest) for person detection.

### 3. Multi-Object Tracking
- Powered by **ByteTrack** from the Supervision library.
- Each detected person receives a **persistent unique ID** that remains stable across frames.

### 4. Contours & Visualization
- Uses **Supervision (sv)** to draw colored **contours** (instead of bounding boxes).
- Each ID gets a **unique, deterministic color** for clear visualization.
- ID labels appear above contours for quick identification.

### 5. Output Logging
| Output Type         | Description                                                             | Location          |
| ------------------- | ----------------------------------------------------------------------- | ----------------- |
| **Annotated Video** | Video with colored contours, tracking IDs, trajectories, and live panel | `runs/annotated/` |
| **Tracking CSV**    | Structured frame-by-frame tracking log                                  | `runs/logs/*_tracks.csv`      |
| **Event CSV**       | Entry and exit event logs                                               | `runs/logs/*_events.csv`      |
| **Snapshots ZIP**   | Cropped images of each unique person                                    | `runs/snapshots/` |

---

## Features

| Feature | Description |
|----------|--------------|
| **Trajectory Tracing** | Visual line showing each person’s movement path. |
| **Entry/Exit Detection** | Logs when new people appear or disappear from frame. |
| **Live Summary Panel** | Overlay showing number of people, active IDs, confidence, entries/exits, and FPS. |
| **Snapshot Export** | Automatically saves cropped images of each unique person on first detection. |
| **Performance Metrics** | Calculates live FPS and overall performance. |

---

## How It Works

1. **Detection** — YOLO model identifies all persons in a frame.
2. **Tracking** — ByteTrack assigns persistent IDs to maintain identity across frames.
3. **Visualization** — Supervision library renders contours and trajectory lines with unique colors.
4. **Logging** — All detection and event data are stored in CSVs for post-analysis.
5. **Dashboard** — Gradio-based interface allows video upload, processing, and result downloads.

---

## Project Structure

```
MultipersonContourTracking/
│
├── data/                      # Input videos
├── runs/
│   ├── annotated/             # Output videos with overlays
│   ├── logs/                  # Tracking & event CSV files
│   └── snapshots/             # Saved cropped person images
├── WorkingDemo.mp4            # Working demo of the code
├── main.py                    # Main Gradio app & processing pipeline
├── requirements.txt           # Python dependencies
├── yolo11n-seg.pt             # YOLOv11 Nano segmentation model
├── yolo11s-seg.pt             # YOLOv11 Small segmentation model
└── yolov8n-seg.pt             # YOLOv8 Nano segmentation model
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MultiPersonContourTracking.git
cd MultiPersonContourTracking
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate   # On Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

### 5. Launch Interface
The Gradio interface will open in your browser at:
```
http://127.0.0.1:7860
```

---

## Output Files

| File Type | Description |
|------------|--------------|
| `.mp4` | Annotated output video with contours, labels, and panel (runs/annotated) |
| `.csv` | Frame-wise tracking data (runs/logs) |
| `.zip` | Cropped snapshots of detected persons (runs/snapshots)|

---

## Assumptions & Limitations

- Designed primarily for **person tracking** (COCO class ID = 0).   
- Slow Processing due to CPU-only systems during segmentation tasks.

---

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Supervision Library](https://github.com/roboflow/supervision)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

---

