import cv2
import os
import random
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
VIDEO_FOLDER = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer"
OUTPUT_POS = r"C:\path\to\dataset\positives"
OUTPUT_NEG = r"C:\path\to\dataset\negatives"

FRAME_STEP = 4
TARGET_W, TARGET_H = 64, 128
NEGATIVE_ATTEMPTS = 5

os.makedirs(OUTPUT_POS, exist_ok=True)
os.makedirs(OUTPUT_NEG, exist_ok=True)

# YOLO model
model = YOLO("yolov8s.pt")

# ---------------- HELPERS ----------------
def boxes_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1 < x2 and y1 < y2

def random_negative_crop(frame, person_boxes):
    h, w, _ = frame.shape
    for _ in range(NEGATIVE_ATTEMPTS):
        x = random.randint(0, max(0, w - TARGET_W))
        y = random.randint(0, max(0, h - TARGET_H))
        candidate = (x, y, x + TARGET_W, y + TARGET_H)
        if not any(boxes_overlap(candidate, b) for b in person_boxes):
            return frame[y:y+TARGET_H, x:x+TARGET_W]
    return None


# ---------------- GLOBAL COUNTERS ----------------
global_pos = 0
global_neg = 0


# ---------------- PROCESS VIDEO ----------------
def process_video(video_path):
    global global_pos, global_neg

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % FRAME_STEP != 0:
            frame_i += 1
            continue

        results = model(frame, verbose=False)
        person_boxes = []

        # YOLO detections
        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # clamp to image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)

                    if x2 > x1 and y2 > y1:
                        person_boxes.append((x1, y1, x2, y2))

        # ---------------- POSITIVES ----------------
        for (x1, y1, x2, y2) in person_boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (TARGET_W, TARGET_H))
            fname = f"pos_{video_id}_{frame_i}_{global_pos}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_POS, fname), crop)
            global_pos += 1

        # ---------------- NEGATIVES (always try) ----------------
        neg_crop = random_negative_crop(frame, person_boxes)
        if neg_crop is not None and neg_crop.size > 0:
            fname = f"neg_{video_id}_{frame_i}_{global_neg}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_NEG, fname), neg_crop)
            global_neg += 1

        frame_i += 1

    cap.release()
    print(f"{video_id}:  pos={global_pos}  neg={global_neg}")


# ---------------- RUN ALL VIDEOS ----------------
for file in os.listdir(VIDEO_FOLDER):
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        full_path = os.path.join(VIDEO_FOLDER, file)
        print(f"Processing {full_path}")
        process_video(full_path)

print("ALL VIDEOS DONE.")
