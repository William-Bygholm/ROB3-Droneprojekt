import cv2
import joblib
import numpy as np
import json
import time
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ---------------- USER INPUT ----------------
VIDEO_IN = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\Civil person.MP4"
MODEL_FILE = "svm_hog_model.pkl_v3"
JSON_COCO = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\Træning\Civil person.json"

WINDOW_SIZE = (128, 256)
SCALES = [1.0, 0.8]
STEP_SIZES = {1.0: 64, 0.8: 56}
FRAME_SKIP = 4
IOU_POSITIVE = 0.5

# ---------------- LOAD MODEL ----------------
clf = joblib.load(MODEL_FILE)
hog = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(8, 8),
    _nbins=9
)

# ---------------- LOAD COCO JSON ----------------
with open(JSON_COCO, "r") as f:
    coco = json.load(f)

# Map image_id -> frame_id (1-baseret)
image_id_to_frame = {img["id"]: idx+1 for idx, img in enumerate(coco["images"])}

# Map frame_id -> list of boxes [x1,y1,x2,y2]
frame_to_boxes = {}
for ann in coco["annotations"]:
    frame_id = image_id_to_frame[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
    frame_to_boxes.setdefault(frame_id, []).append([x1, y1, x2, y2])

# ---------------- VIDEO INFO ----------------
cap_tmp = cv2.VideoCapture(VIDEO_IN)
ret, frame_tmp = cap_tmp.read()
if not ret:
    raise ValueError("Could not read video frame!")
video_h, video_w = frame_tmp.shape[:2]
cap_tmp.release()

# COCO billedopløsning (første billede)
coco_w, coco_h = coco["images"][0]["width"], coco["images"][0]["height"]
scale_x = video_w / coco_w
scale_y = video_h / coco_h
print(f"Video resolution: {video_w}x{video_h}, COCO: {coco_w}x{coco_h}")
print(f"Scaling factors - X: {scale_x:.4f}, Y: {scale_y:.4f}")

# ---------------- HELPERS ----------------
def sliding_windows(img, step, win_size):
    w, h = win_size
    for y in range(0, img.shape[0]-h+1, step):
        for x in range(0, img.shape[1]-w+1, step):
            yield x, y, img[y:y+h, x:x+w]

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

# ---------------- RUN DETECTION ----------------
scores = []
labels = []

cap = cv2.VideoCapture(VIDEO_IN)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_id = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape[:2]

    # Ground-truth for denne frame, skaleret
    gt_boxes = frame_to_boxes.get(frame_id, [])
    gt_boxes_scaled = [[int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)] for x1,y1,x2,y2 in gt_boxes]

    for scale in SCALES:
        resized = cv2.resize(gray, None, fx=scale, fy=scale)
        step = STEP_SIZES[scale]
        scale_x_win = w0 / resized.shape[1]
        scale_y_win = h0 / resized.shape[0]

        for x, y, win in sliding_windows(resized, step, WINDOW_SIZE):
            if win.shape != (WINDOW_SIZE[1], WINDOW_SIZE[0]):
                continue

            feat = hog.compute(win).ravel()
            score = clf.decision_function([feat])[0]

            x1 = int(x * scale_x_win)
            y1 = int(y * scale_y_win)
            x2 = int((x + WINDOW_SIZE[0]) * scale_x_win)
            y2 = int((y + WINDOW_SIZE[1]) * scale_y_win)
            box = [x1, y1, x2, y2]

            # IoU match
            label = 0
            for g in gt_boxes_scaled:
                if iou(box, g) > IOU_POSITIVE:
                    label = 1
                    break

            scores.append(score)
            labels.append(label)

    # Progress
    elapsed = time.time() - start_time
    progress = frame_id / total_frames
    if progress > 0:
        est_total = elapsed / progress
        remaining = est_total - elapsed
        print(f"Frame {frame_id}/{total_frames} "
              f"({progress*100:.2f}%), "
              f"Elapsed: {elapsed:.1f}s, "
              f"Remaining: {remaining:.1f}s", end="\r")

cap.release()
print("\nVideo processing complete.")

scores = np.array(scores)
labels = np.array(labels)

# ---------------- PLOT ROC + PR ----------------
fpr, tpr, _ = roc_curve(labels, scores)
prec, rec, _ = precision_recall_curve(labels, scores)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.4f}")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(rec, prec, label=f"PR AUC = {auc(rec, prec):.4f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.show()

print("AUC ROC:", auc(fpr, tpr))
print("AUC PR:", auc(rec, prec))
