from ultralytics import YOLO
from pathlib import Path 
import cv2
import glob
import os

# --- SETTINGS ---
image_folder = "Billeder"   # mappen hvor billeder ligger
output_folder = "Output"    # YOLO-resultater
process_video = True        # set True to process a video in Billeder, False to process images
video_name = "Elias walking - Copy.mp4"  # change to your video filename if needed

os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

if process_video:
    video_path = Path("Billeder") / video_name
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- scale output to a maximum dimension (preserves aspect ratio) ---
    max_output_dim = 720  # change this to your desired max width/height
    scale = min(1.0, max_output_dim / max(w, h))
    out_w, out_h = int(w * scale), int(h * scale)

    out_path = Path(output_folder) / (video_path.stem + "_yolo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    print(f"Processing video {video_path} -> {out_path} (fps={fps}, input={w}x{h}, output={out_w}x{out_h})")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # run YOLO on the full-size frame (better accuracy)
        results = model(frame)[0]

        # draw only people (class 0)
        for box in results.boxes:
            try:
                cls = int(box.cls[0])
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            except Exception:
                vals = box.xyxy
                if vals is not None and len(vals) > 0:
                    x1, y1, x2, y2 = map(int, vals[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # resize the annotated frame for writing/display (keeps detection on original)
        if scale < 1.0:
            out_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            out_frame = frame

        writer.write(out_frame)
        cv2.imshow("YOLO Video", out_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("Done. Saved:", out_path)

else:
    # --- original image folder mode ---
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))

    print(f"Fandt {len(image_paths)} billeder")

    for path in image_paths:
        img = cv2.imread(path)
        results = model(img)[0]

        # Tegn kun mennesker (class 0)
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Gem output-billedet
        filename = os.path.basename(path)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, img)

        # Vis det
        cv2.imshow("YOLO Result", img)
        cv2.waitKey(300)

    cv2.destroyAllWindows()
