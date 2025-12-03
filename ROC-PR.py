import os
import time
import json
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score

# ---------------- USER INPUT ----------------
# Multiple video/JSON pairs for testing
TEST_DATASETS = [
    {
        'video': r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4",
        'json': r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\Validation\2 mili med blå bond.json"
    }
]

# HOG window and scanning parameters
WINDOW_SIZE = (128, 256)  # (width, height)
SCALES = [1.0, 0.8]       # pyramid scales
STEP_SIZES = {1.0: 64, 0.8: 56}
FRAME_SKIP = 1
IOU_POSITIVE = 0.5

# ---------------- HELPERS ----------------
def sliding_windows(img, step, win_size):
    w, h = win_size
    H, W = img.shape[:2]
    if H < h or W < w:
        return  # no windows if image smaller than win size
    for y in range(0, H - h + 1, step):
        for x in range(0, W - w + 1, step):
            yield x, y, img[y:y+h, x:x+w]

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0, (a[2] - a[0])) * max(0, (a[3] - a[1]))
    area_b = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
    denom = (area_a + area_b - inter)
    return inter / (denom + 1e-9)

def safe_read_frame(cap):
    ret, frame = cap.read()
    if not ret or frame is None:
        return False, None
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0 or h > 8000 or w > 8000:
        # guard against bogus dimensions
        return False, None
    return True, frame

def maybe_downscale(frame, max_w=None, max_h=None):
    if max_w is None and max_h is None:
        return frame
    h, w = frame.shape[:2]
    scale_w = 1.0 if max_w is None else min(1.0, max_w / float(w))
    scale_h = 1.0 if max_h is None else min(1.0, max_h / float(h))
    scale = min(scale_w, scale_h)
    if scale >= 1.0:
        return frame
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ---------------- LOAD MODEL ----------------
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

model_data = joblib.load(MODEL_FILE)
clf = model_data['classifier'] if isinstance(model_data, dict) and 'classifier' in model_data else model_data

hog = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(8, 8),
    _nbins=9
)

# ---------------- CHECK INPUTS ----------------
if not os.path.exists(VIDEO_IN):
    raise FileNotFoundError(f"Video not found: {VIDEO_IN}")
if not os.path.exists(JSON_COCO):
    raise FileNotFoundError(f"JSON not found: {JSON_COCO}")

# ---------------- LOAD COCO JSON ----------------
with open(JSON_COCO, "r") as f:
    coco = json.load(f)

if "images" not in coco or "annotations" not in coco or len(coco["images"]) == 0:
    raise ValueError("Invalid COCO JSON: missing 'images' or 'annotations'")

# Map image_id -> frame_id (1-based by enumeration)
image_id_to_frame = {img["id"]: idx + 1 for idx, img in enumerate(coco["images"])}

# Map frame_id -> list of boxes [x1,y1,x2,y2] in COCO resolution
frame_to_boxes = {}
for ann in coco["annotations"]:
    frame_id = image_id_to_frame.get(ann.get("image_id"))
    if frame_id is None:
        continue
    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    frame_to_boxes.setdefault(frame_id, []).append([x1, y1, x2, y2])

# ---------------- VIDEO INFO + SCALING ----------------
cap_tmp = cv2.VideoCapture(VIDEO_IN)
ok, frame_tmp = safe_read_frame(cap_tmp)
if not ok:
    cap_tmp.release()
    raise RuntimeError(f"Could not read a valid frame from: {VIDEO_IN}")

# Apply optional downscale to determine working resolution
frame_tmp = maybe_downscale(frame_tmp, DOWNSCALE_MAX_W, DOWNSCALE_MAX_H)
video_h, video_w = frame_tmp.shape[:2]
cap_tmp.release()

# COCO resolution (use first image entry)
coco_w = int(coco["images"][0]["width"])
coco_h = int(coco["images"][0]["height"])
if coco_w <= 0 or coco_h <= 0:
    raise ValueError("Invalid COCO image width/height")

scale_x_coco_to_video = video_w / float(coco_w)
scale_y_coco_to_video = video_h / float(coco_h)

print(f"[INFO] Video resolution: {video_w}x{video_h}, COCO: {coco_w}x{coco_h}")
print(f"[INFO] Scaling factors - X: {scale_x_coco_to_video:.4f}, Y: {scale_y_coco_to_video:.4f}")

# ---------------- RUN DETECTION ----------------
scores = []
labels = []

cap = cv2.VideoCapture(VIDEO_IN)
total_frames = int(max(1, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
frame_id = 0
start_time = time.time()

while True:
    ok, frame = safe_read_frame(cap)
    if not ok:
        break

    frame_id += 1
    if MAX_FRAMES is not None and frame_id > MAX_FRAMES:
        break
    if frame_id % FRAME_SKIP != 0:
        continue

    # Optional downscale for memory
    frame = maybe_downscale(frame, DOWNSCALE_MAX_W, DOWNSCALE_MAX_H)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape[:2]

    # Ground-truth for this frame, scaled to working video resolution
    gt_boxes = frame_to_boxes.get(frame_id, [])
    gt_boxes_scaled = [
        [
            int(x1 * scale_x_coco_to_video),
            int(y1 * scale_y_coco_to_video),
            int(x2 * scale_x_coco_to_video),
            int(y2 * scale_y_coco_to_video),
        ]
        for (x1, y1, x2, y2) in gt_boxes
    ]

    for scale in SCALES:
        # Resize gray for scanning at this pyramid level
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        step = STEP_SIZES.get(scale, 64)

        Hs, Ws = resized.shape[:2]
        # Map window coordinates back to original working resolution
        scale_x_win = w0 / float(Ws)
        scale_y_win = h0 / float(Hs)

        for x, y, win in sliding_windows(resized, step, WINDOW_SIZE):
            if win.shape != (WINDOW_SIZE[1], WINDOW_SIZE[0]):
                continue

            feat = hog.compute(win).ravel()
            # Decision function expects 2D array
            score = clf.decision_function([feat])[0]

            x1 = int(x * scale_x_win)
            y1 = int(y * scale_y_win)
            x2 = int((x + WINDOW_SIZE[0]) * scale_x_win)
            y2 = int((y + WINDOW_SIZE[1]) * scale_y_win)
            box = [x1, y1, x2, y2]

            # IoU match to any GT box
            label = 0
            for g in gt_boxes_scaled:
                if iou(box, g) > IOU_POSITIVE:
                    label = 1
                    break

            scores.append(score)
            labels.append(label)

    # Progress
    elapsed = time.time() - start_time
    progress = frame_id / max(1, total_frames)
    if progress > 0:
        est_total = elapsed / progress
        remaining = max(0.0, est_total - elapsed)
        print(
            f"Frame {frame_id}/{total_frames} ({progress*100:.2f}%), "
            f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s      ",
            end="\r"
        )

cap.release()
print(f"\n[INFO] Processing complete. Collected {len(scores)} detections.")

# ---------------- METRICS & THRESHOLD ----------------
scores = np.array(scores, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

if len(scores) == 0:
    raise RuntimeError("No detections collected. Check model, window size, or video/JSON alignment.")

pos_count = int(labels.sum())
neg_count = int(len(labels) - pos_count)
print(f"[INFO] Total detections: {len(scores)} | Positives: {pos_count} | Negatives: {neg_count}")

# Find best threshold by F1
thresholds = np.linspace(scores.min(), scores.max(), 100)
f1_scores = []
for thr in thresholds:
    preds = (scores >= thr).astype(int)
    f1_scores.append(f1_score(labels, preds))

best_idx = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])
best_f1 = float(f1_scores[best_idx])

# Metrics at best threshold
best_predictions = (scores >= best_threshold).astype(int)
tp = int(np.sum((labels == 1) & (best_predictions == 1)))
fp = int(np.sum((labels == 0) & (best_predictions == 1)))
fn = int(np.sum((labels == 1) & (best_predictions == 0)))
tn = int(np.sum((labels == 0) & (best_predictions == 0)))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
accuracy = (tp + tn) / float(len(labels))

print("\n" + "=" * 60)
print("OPTIMAL THRESHOLD ANALYSIS")
print("=" * 60)
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")
print("\nMetrics at best threshold:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Accuracy:  {accuracy:.4f}")
print("\nConfusion Matrix:")
print(f"  TP: {tp:6d}  FP: {fp:6d}")
print(f"  FN: {fn:6d}  TN: {tn:6d}")
print("=" * 60)

# ---------------- PLOT ROC + PR ----------------
fpr, tpr, roc_thresholds = roc_curve(labels, scores)
prec, rec, pr_thresholds = precision_recall_curve(labels, scores)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
# highlight operating point at best threshold
fpr_pt = fp / max(1, (fp + tn))
tpr_pt = tp / max(1, (tp + fn))
plt.scatter([fpr_pt], [tpr_pt], color='red', s=100, zorder=5, label=f'Best thr = {best_threshold:.4f}')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(rec, prec, label=f"AUC = {auc(rec, prec):.4f}")
plt.scatter([recall], [precision], color='red', s=100, zorder=5, label=f'Best thr = {best_threshold:.4f}')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.show()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"AUC ROC: {auc(fpr, tpr):.4f}")
print(f"AUC PR:  {auc(rec, prec):.4f}")
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best F1 Score:  {best_f1:.4f}")
print("=" * 60)
