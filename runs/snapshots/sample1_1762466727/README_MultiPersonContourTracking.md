# Advanced Multi-Person Contour Tracking Dashboard

## 📘 Project Overview
This project implements a **state-of-the-art computer vision pipeline** for detecting, tracking, and visualizing multiple people in video feeds using open-source tools like **YOLO**, **Supervision**, and **Roboflow**.

The system is designed to perform **real-time person tracking** with persistent IDs, contour visualization, and structured data logging — suitable for real-world applications such as surveillance, retail analytics, or crowd management.

---

## 🎯 Core Features

### 1. Video Ingestion
- Accepts both **recorded video files** and **live webcam feeds**.
- Handles standard formats like `.mp4`, `.avi`, etc.

### 2. Entity Detection
- Uses **YOLO-based models** (`YOLOv8`, `YOLO11`) for person detection.
- Custom model integration supported via **Roboflow**.
- Configurable confidence threshold for detections.

### 3. Multi-Object Tracking
- Powered by **ByteTrack** from the Supervision library.
- Each detected person receives a **persistent unique ID** that remains stable across frames.

### 4. Contours & Visualization
- Uses **Supervision (sv)** to draw colored **contours** (instead of bounding boxes).
- Each ID gets a **unique, deterministic color** for clear visualization.
- ID labels appear above contours for quick identification.

### 5. Output Logging
- Per-frame data stored in `.csv` format under `runs/logs/`.
- Each record includes:
  - Frame index
  - Timestamp
  - Track ID
  - Bounding box coordinates `(x1, y1, x2, y2)`
  - Confidence score

---

## 🌟 Stretch Features

| Feature | Description |
|----------|--------------|
| **Trajectory Tracing** | Visual line showing each person’s movement path. |
| **Entry/Exit Detection** | Logs when new people appear or disappear from frame. |
| **Live Summary Panel** | Real-time overlay showing number of people, active IDs, confidence, entries/exits, and FPS. |
| **Snapshot Export** | Automatically saves cropped images of each unique person on first detection. |
| **Performance Metrics** | Calculates live FPS and overall performance. |

---

## 🧠 How It Works

1. **Detection** — YOLO model identifies all persons in a frame.
2. **Tracking** — ByteTrack assigns persistent IDs to maintain identity across frames.
3. **Visualization** — Supervision library renders contours and trajectory lines with unique colors.
4. **Logging** — All detection and event data are stored in CSVs for post-analysis.
5. **Dashboard** — Gradio-based interface allows video upload, processing, and result downloads.

---

## 🧩 Project Structure

```
MultipersonContourTracking/
│
├── data/                      # Input videos
├── runs/
│   ├── annotated/             # Output videos with overlays
│   ├── logs/                  # Tracking & event CSV files
│   └── snapshots/             # Saved cropped person images
│
├── main.py                    # Main Gradio app & processing pipeline
├── requirements.txt           # Python dependencies
├── yolo11n-seg.pt             # YOLOv11 Nano segmentation model
├── yolo11s-seg.pt             # YOLOv11 Small segmentation model
└── yolov8n-seg.pt             # YOLOv8 Nano segmentation model
```

---

## ⚙️ Installation & Setup

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

## 📊 Output Files

| File Type | Description |
|------------|--------------|
| `.mp4` | Annotated output video with contours, labels, and panel |
| `.csv` | Frame-wise tracking data (runs/logs) |
| `.zip` | Cropped snapshots of detected persons |

---

## 🧩 Assumptions & Limitations

- Designed primarily for **person tracking** (COCO class ID = 0).  
- Tracking IDs may occasionally switch due to **occlusion** or **rapid motion**.  
- Real-time performance depends on GPU availability and input video resolution.  
- FPS may drop on CPU-only systems during segmentation tasks.

---

## 📘 References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com)
- [Supervision Library](https://github.com/roboflow/supervision)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

---

## 🏁 Author
**Developed by:** [Your Name]  
**Course / Project:** Multi-Person Contour Tracking - Vision AI Pipeline  
**Institution:** [Your Institution Name]

