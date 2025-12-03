import json
import cv2
import joblib
from skimage.feature import hog
import numpy as np
from time import time

# ============================================
# CONFIG
# ============================================

pairs = [
    (
        r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 mili en idiot der ligger ned.MP4",
        r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json"
    ),
    (
        r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\3 mili 2 onde 1 god.MP4",
        r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\3mili 2 onde 1 god.json"
    )
]

print("Loader model...")
clf, winW, winH = joblib.load(r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\person_detector_trained.pkl")
print("Model loaded:", (winW, winH))


# ============================================
# Helpers
# ============================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sliding_window(image, step, window_size):
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def extract_hog(img):
    return hog(img,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               block_norm='L2-Hys',
               transform_sqrt=True)


# ============================================
# MAIN: Loop igennem video + json par
# ============================================
for (video_path, json_path) in pairs:

    print("\n=== Evaluating:", video_path, "===")
    annotations = load_json(json_path)

    # OPTIONAL: find alle ground truth bokse pr. frame (det er let i COCO)
    gt_by_frame = {}
    if "annotations" in annotations:
        for ann in annotations["annotations"]:
            fid = ann["image_id"]
            if fid not in gt_by_frame:
                gt_by_frame[fid] = []
            gt_by_frame[fid].append(ann["bbox"])
    # hvis din version ikke har annotations, så er GT tom → no problem

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Kan ikke åbne video:", video_path)
        continue

    frame_idx = 0
    stride = 16
    scale_factor = 1.20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_t = time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = []
        scaled = gray.copy()
        scale = 1.0

        while scaled.shape[0] >= winH and scaled.shape[1] >= winW:
            for (x, y, window) in sliding_window(scaled, stride, (winW, winH)):
                hog_feat = extract_hog(window)
                pred = clf.predict([hog_feat])[0]

                if pred == 1:
                    rx = int(x * (1/scale))
                    ry = int(y * (1/scale))
                    rw = int(winW * (1/scale))
                    rh = int(winH * (1/scale))
                    detections.append((rx, ry, rw, rh))

            scale *= scale_factor
            newW = int(gray.shape[1] / scale)
            newH = int(gray.shape[0] / scale)
            scaled = cv2.resize(gray, (newW, newH))

        # tegn detection
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # tegn ground truth hvis du vil
        if frame_idx in gt_by_frame:
            for bbox in gt_by_frame[frame_idx]:
                gx, gy, gw, gh = map(int, bbox)
                cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

        fps = 1.0 / (time() - start_t)
        print(f"Frame {frame_idx}: {fps:.2f} FPS")

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()

cv2.destroyAllWindows()
