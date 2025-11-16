import cv2
import os
import random
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
VIDEO_FOLDER = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer"          # mappe med dine videoer
OUTPUT_POS = r"C:\path\to\dataset\positives"
OUTPUT_NEG = r"C:\path\to\dataset\negatives"

FRAME_STEP = 4
TARGET_W, TARGET_H = 64, 128   # HOG standard size
NEGATIVE_ATTEMPTS = 5          # hvor mange forsøg på at finde non-overlap crop

os.makedirs(OUTPUT_POS, exist_ok=True)
os.makedirs(OUTPUT_NEG, exist_ok=True)

# YOLO model
model = YOLO("yolov8s.pt")     # lille og hurtig, godt til autolabeling

# ---------------- HELPER ----------------
def boxes_overlap(box1, box2):
    """Return True if boxes overlap. Boxes = (x1,y1,x2,y2)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1 < x2 and y1 < y2

def random_negative_crop(frame, person_boxes):
    """Find a random 64x128 crop that doesn't overlap with any person"""
    h, w, _ = frame.shape
    for _ in range(NEGATIVE_ATTEMPTS):
        x = random.randint(0, max(0, w - TARGET_W))
        y = random.randint(0, max(0, h - TARGET_H))
        candidate = (x, y, x + TARGET_W, y + TARGET_H)
        if not any(boxes_overlap(candidate, b) for b in person_boxes):
            return frame[y:y+TARGET_H, x:x+TARGET_W]
    return None  # hvis ikke fundet efter NEGATIVE_ATTEMPTS

# ---------------- PROCESS VIDEO ----------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    frame_i = 0
    save_i_pos = 0
    save_i_neg = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % FRAME_STEP != 0:
            frame_i += 1
            continue

        results = model(frame, verbose=False)
        person_boxes = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                if cls == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))

        # ------------- POSITIVES ----------------
        for box in person_boxes:
            x1, y1, x2, y2 = box
            crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop, (TARGET_W, TARGET_H))
            save_path = os.path.join(OUTPUT_POS, f"pos_{save_i_pos}.jpg")
            cv2.imwrite(save_path, resized)
            save_i_pos += 1

        # ------------- NEGATIVES ----------------
        if person_boxes:
            neg_crop = random_negative_crop(frame, person_boxes)
            if neg_crop is not None:
                save_path = os.path.join(OUTPUT_NEG, f"neg_{save_i_neg}.jpg")
                cv2.imwrite(save_path, neg_crop)
                save_i_neg += 1

        frame_i += 1

    cap.release()
    print(f"Done: {video_path}, positives: {save_i_pos}, negatives: {save_i_neg}")

# ---------------- RUN ALL VIDEOS ----------------
for file in os.listdir(VIDEO_FOLDER):
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        full_path = os.path.join(VIDEO_FOLDER, file)
        print(f"Processing {full_path}")
        process_video(full_path)

print("ALL VIDEOS DONE.")
