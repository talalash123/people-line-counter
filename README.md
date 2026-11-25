# People Line Counter

A real-time computer vision pipeline for detecting, tracking, and counting individuals crossing a virtual boundary line in video streams. Built using a custom-trained YOLOv8 detection model, ByteTrack multi-object tracking, and OpenCV.

## Overview

People Line Counter processes video input frame by frame to identify people, maintain persistent tracking IDs across consecutive frames, render movement trajectories, and increment directional entry (`IN`) and exit (`OUT`) counters whenever a tracked person crosses a configurable horizontal boundary line.

## Key Features

- **Custom Object Detection:** Uses a custom-trained YOLOv8 model (`best.pt`) to detect people across video frames.
- **Persistent Multi-Object Tracking:** Integrates ByteTrack (`bytetrack.yaml`) to maintain consistent track IDs across frames.
- **Directional Line Crossing:** Calculates bounding-box centroids and compares frame-to-frame vertical positions against a virtual horizontal line to classify `IN` (top-to-bottom) and `OUT` (bottom-to-top) crossings.
- **Duplicate Prevention:** Stores counted track IDs in memory to avoid repeated counts when an individual pauses or oscillates near the counting boundary.
- **Trajectory and HUD Visualization:** Draws historical movement trails, bounding boxes, centroid markers, and a live HUD displaying `IN`, `OUT`, and `ACTIVE` counts.
- **Reliable Video Export:** Enforces even frame dimensions and selects compatible codecs (`XVID` / `MJPG`) to export playable `.avi` output videos.

## Project Structure

```text
people-line-counter/
├── best.pt          # Custom-trained YOLOv8 weights
├── counter.py       # Main tracking, line-crossing, and video export script
├── test.mp4         # Input video file (ignored in Git)
├── output.avi       # Processed output video (ignored in Git)
├── .gitignore       # Excludes local media files and Python cache
└── README.md        # Project documentation
```

## Requirements

- Python 3.8 or higher
- `ultralytics`
- `opencv-python`

Install the required dependencies:

```bash
pip install ultralytics opencv-python
```

## Configuration

Adjustable parameters at the top of `counter.py`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `"best.pt"` | Path to the trained YOLO model weights |
| `VIDEO_PATH` | `"test.mp4"` | Path to the input video file |
| `OUTPUT_PATH` | `"output.avi"` | Path for the exported annotated video |
| `LINE_RATIO` | `0.55` | Vertical position of the counting line (`0.0` top to `1.0` bottom) |
| `CONF_THRESHOLD` | `0.35` | Minimum detection confidence threshold |
| `TRAIL_LENGTH` | `20` | Number of previous centroid points stored per trajectory trail |
| `SHOW_PREVIEW` | `False` | Toggle real-time OpenCV preview window (requires GUI-enabled OpenCV) |

## Usage

1. Place your trained weights (`best.pt`) and input video (`test.mp4`) in the project root directory.
2. Run the counting pipeline:

```bash
python counter.py
```

3. Once processing completes, open `output.avi` to inspect bounding boxes, track IDs, movement trails, and the `IN` / `OUT` counters.

## How the Counting Logic Works

1. **Detection and Tracking:** Each video frame is passed to `model.track(..., persist=True, tracker="bytetrack.yaml")` to obtain bounding boxes and persistent `track_id` values.
2. **Centroid Calculation:** For every tracked bounding box `(x1, y1, x2, y2)`, the center point `(cx, cy)` is computed.
3. **Boundary Crossing Check:** Each track ID's previous vertical coordinate (`prev_y`) is compared against its current coordinate (`cy`) relative to `line_y`:
   - `prev_y < line_y` and `cy >= line_y` increments `IN`.
   - `prev_y > line_y` and `cy <= line_y` increments `OUT`.
4. **State Locking:** Once a `track_id` triggers a crossing, it is added to `counted_ids` so it cannot be counted multiple times during the same track.

## Limitations and Future Improvements

- **Single Horizontal Line:** Currently supports one horizontal counting line; future updates can add angled lines or polygonal entry/exit zones.
- **Re-entry Handling:** Because `counted_ids` locks an ID permanently for the session, an individual who exits and later re-enters under the same ID will not be recounted unless cooldown-based state resets are added.
- **Occlusion and ID Switches:** Heavy crowd overlap can cause tracker ID resets; tuning ByteTrack parameters or using a higher-capacity backbone can improve stability in dense scenes.
