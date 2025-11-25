# dataset_builder.py
import os
import cv2
import numpy as np 
import random 
from tqdm import tqdm
from ultralytics import YOLO

# ---------------- SETTINGS ----------------
VIDEO_FOLDER = r"C:\Users\alexa\Desktop\ProjektVideoer"
OUTPUT_POS = r"C:\Users\alexa\Desktop\Pos"
OUTPUT_NEG = r"C:\Users\alexa\Desktop\Neg"

FRAME_STEP = 2          # Process every 2nd frame
TARGET_SIZE = 128       # Square HOG window
NEG_PER_FRAME = 5       # Number of negatives per frame
POS_PADDING = 0.3       # 30% padding around person

os.makedirs(OUTPUT_POS, exist_ok=True)
os.makedirs(OUTPUT_NEG, exist_ok=True)

# ---------------- YOLO ----------------
model = YOLO("yolov8s.pt")  # CPU by default

# ---------------- HELPERS ----------------
def boxes_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1 < x2 and y1 < y2

def safe_crop(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return frame[y1:y2, x1:x2]

def find_negative_crops(frame, person_boxes, max_neg=NEG_PER_FRAME):
    h, w, _ = frame.shape
    negatives = []
    attempts = 0
    while len(negatives) < max_neg and attempts < 100:
        x = random.randint(0, max(0, w - TARGET_SIZE))
        y = random.randint(0, max(0, h - TARGET_SIZE))
        cand = (x, y, x + TARGET_SIZE, y + TARGET_SIZE)
        if not any(boxes_overlap(cand, b) for b in person_boxes):
            negatives.append(frame[y:y+TARGET_SIZE, x:x+TARGET_SIZE])
        attempts += 1
    return negatives

global_pos, global_neg = 0, 0

# ---------------- PROCESS VIDEO ----------------
def process_video(video_path):
    global global_pos, global_neg
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_i % FRAME_STEP != 0:
            frame_i += 1
            continue

        # ---------------- YOLO PREDICTION ----------------
        results = model(frame, device="cpu", verbose=False)
        person_boxes = []

        for r in results:
            for box in r.boxes:
                if int(box.cls) == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w_box = x2 - x1
                    h_box = y2 - y1
                    pad_w = int(w_box * POS_PADDING)
                    pad_h = int(h_box * POS_PADDING)

                    x1_pad = x1 - pad_w
                    y1_pad = y1 - pad_h
                    x2_pad = x2 + pad_w
                    y2_pad = y2 + pad_h

                    crop = safe_crop(frame, x1_pad, y1_pad, x2_pad, y2_pad)
                    if crop.size == 0:
                        continue

                    crop_resized = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE))
                    fname = f"pos_{video_id}_{frame_i}_{global_pos}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_POS, fname), crop_resized)
                    global_pos += 1

                    # Store padded box for negative avoidance
                    person_boxes.append((x1_pad, y1_pad, x2_pad, y2_pad))

        # ---------------- NEGATIVES ----------------
        neg_crops = find_negative_crops(frame, person_boxes, max_neg=NEG_PER_FRAME)
        for nc in neg_crops:
            fname = f"neg_{video_id}_{frame_i}_{global_neg}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_NEG, fname), nc)
            global_neg += 1

        if frame_i % 50 == 0 or frame_i == total_frames - 1:
            print(f"[{video_id}] Frame {frame_i}/{total_frames} — pos={global_pos}, neg={global_neg}")

        frame_i += 1

    cap.release()
    print(f"[{video_id}] DONE — total pos={global_pos}, neg={global_neg}")

# ---------------- RUN ----------------
for file in os.listdir(VIDEO_FOLDER):
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(os.path.join(VIDEO_FOLDER, file))

print("ALL VIDEOS DONE.")
print("FINAL COUNT — POS:", global_pos, "NEG:", global_neg)
