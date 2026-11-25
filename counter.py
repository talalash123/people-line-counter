from collections import defaultdict, deque
import cv2
from ultralytics import YOLO

# File paths
MODEL_PATH = "best.pt"
VIDEO_PATH = "test.mp4"
OUTPUT_PATH = "output.avi"

# Virtual horizontal counting line ratio (0.55 = slightly below vertical center)
LINE_RATIO = 0.55
CONF_THRESHOLD = 0.35
TRAIL_LENGTH = 20

# Set False when OpenCV GUI/imshow is unavailable so all frames write cleanly
SHOW_PREVIEW = False


def init_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Initialize VideoWriter using reliable codecs."""
    for codec in ("XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"VideoWriter ready with codec '{codec}' -> {output_path}")
            return writer
        writer.release()
    raise RuntimeError("Unable to initialize VideoWriter with XVID or MJPG.")


def main() -> None:
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {VIDEO_PATH}")

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 1.0 or fps > 120.0:
        fps = 25.0

    # Keep even dimensions for clean video encoding
    width = raw_w if raw_w % 2 == 0 else raw_w - 1
    height = raw_h if raw_h % 2 == 0 else raw_h - 1
    line_y = int(height * LINE_RATIO)

    out = init_writer(OUTPUT_PATH, fps, width, height)

    prev_y_map = {}
    counted_ids = set()
    track_trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

    in_count = 0
    out_count = 0
    frames_written = 0

    print(
        f"Processing '{VIDEO_PATH}' | Size: {width}x{height} | "
        f"FPS: {fps:.2f} | Total frames: {total_frames}"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        # Run YOLO detection + ByteTrack tracking
        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # Draw virtual horizontal counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "COUNTING LINE",
            (width - 175, max(22, line_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        active_people = 0

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()

            active_people = len(track_ids)

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Update movement trail
                track_trails[track_id].append((cx, cy))

                # Line-crossing direction logic (counted once per ID)
                if track_id in prev_y_map and track_id not in counted_ids:
                    prev_y = prev_y_map[track_id]

                    if prev_y < line_y and cy >= line_y:
                        in_count += 1
                        counted_ids.add(track_id)
                    elif prev_y > line_y and cy <= line_y:
                        out_count += 1
                        counted_ids.add(track_id)

                prev_y_map[track_id] = cy

                # Draw trajectory trail
                pts = list(track_trails[track_id])
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (255, 180, 0), 2)

                # Draw bounding box, center dot, and ID label
                box_color = (0, 255, 0) if track_id not in counted_ids else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"ID:{track_id} ({conf:.2f})",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                )

        # Top HUD panel
        cv2.rectangle(frame, (12, 12), (430, 62), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"IN: {in_count}   OUT: {out_count}   ACTIVE: {active_people}",
            (22, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        out.write(frame)
        frames_written += 1

        if frames_written % 50 == 0 or frames_written == total_frames:
            print(f"Progress: {frames_written}/{total_frames} frames written...")

        if SHOW_PREVIEW:
            cv2.imshow("People Counter - CV Pipeline", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("Preview stopped early via 'q'.")
                break

    cap.release()
    out.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    duration = frames_written / fps if fps > 0 else 0.0
    print(
        f"Finished! Written frames: {frames_written}/{total_frames} "
        f"(~{duration:.2f}s) -> {OUTPUT_PATH} | IN: {in_count}, OUT: {out_count}"
    )


if __name__ == "__main__":
    main()