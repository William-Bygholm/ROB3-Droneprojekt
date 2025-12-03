import cv2
import joblib
import numpy as np
import json
import time
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

# ---------------- USER INPUT ----------------
# Multiple video/JSON pairs for testing
TEST_DATASETS = [
    {
        'video': r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 mili med blå bånd .MP4",
        'json': r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\Validation\2 mili med blå bond.json"
    }
]

MODEL_FILE = "Person_Detector_Json+YOLO.pkl"

WINDOW_SIZE = (128, 256)
SCALES = [1.0, 0.8]
STEP_SIZES = {1.0: 64, 0.8: 56}
FRAME_SKIP = 1
IOU_POSITIVE = 0.5

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

# ---------------- LOAD MODEL ----------------
model_data = joblib.load(MODEL_FILE)
if isinstance(model_data, dict):
    clf = model_data['classifier']
else:
    clf = model_data

hog = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(8, 8),
    _nbins=9
)

# ---------------- PROCESS MULTIPLE DATASETS ----------------
all_scores = []
all_labels = []

for dataset_idx, dataset in enumerate(TEST_DATASETS, 1):
    VIDEO_IN = dataset['video']
    JSON_COCO = dataset['json']
    
    print(f"\n{'='*60}")
    print(f"Processing dataset {dataset_idx}/{len(TEST_DATASETS)}")
    print(f"Video: {VIDEO_IN}")
    print(f"JSON: {JSON_COCO}")
    print('='*60)
    
    # Check files exist
    if not os.path.exists(VIDEO_IN):
        print(f"WARNING: Video not found, skipping: {VIDEO_IN}")
        continue
    if not os.path.exists(JSON_COCO):
        print(f"WARNING: JSON not found, skipping: {JSON_COCO}")
        continue
    
    # ---------------- LOAD COCO JSON ----------------
    with open(JSON_COCO, "r") as f:
        coco = json.load(f)

    # Map image_id -> frame_id (1-baseret)
    image_id_to_frame = {img["id"]: idx+1 for idx, img in enumerate(coco["images"])}

    # Map frame_id -> list of boxes [x1,y1,x2,y2]
    frame_to_boxes = {}
    for ann in coco["annotations"]:
        frame_id = image_id_to_frame.get(ann["image_id"])
        if frame_id is None:
            continue
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
        frame_to_boxes.setdefault(frame_id, []).append([x1, y1, x2, y2])

    # ---------------- VIDEO INFO ----------------
    cap_tmp = cv2.VideoCapture(VIDEO_IN)
    ret, frame_tmp = cap_tmp.read()
    if not ret:
        print(f"ERROR: Could not read video frame from {VIDEO_IN}")
        cap_tmp.release()
        continue
    video_h, video_w = frame_tmp.shape[:2]
    cap_tmp.release()

    # COCO billedopløsning (første billede)
    coco_w, coco_h = coco["images"][0]["width"], coco["images"][0]["height"]
    scale_x = video_w / coco_w
    scale_y = video_h / coco_h
    print(f"Video resolution: {video_w}x{video_h}, COCO: {coco_w}x{coco_h}")
    print(f"Scaling factors - X: {scale_x:.4f}, Y: {scale_y:.4f}")

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
    print(f"\nVideo {dataset_idx} processing complete. Collected {len(scores)} detections.")
    
    # Add to combined results
    all_scores.extend(scores)
    all_labels.extend(labels)

# ---------------- COMBINED RESULTS ----------------
print(f"\n{'='*60}")
print(f"COMBINED RESULTS FROM ALL DATASETS")
print(f"Total detections: {len(all_scores)}")
print(f"Positives: {sum(all_labels)}, Negatives: {len(all_labels) - sum(all_labels)}")
print('='*60)

scores = np.array(all_scores)
labels = np.array(all_labels)

# ---------------- FIND BEST THRESHOLD ----------------
from sklearn.metrics import f1_score

# Calculate F1 score for different thresholds
thresholds = np.linspace(scores.min(), scores.max(), 100)
f1_scores = []

for threshold in thresholds:
    predictions = (scores >= threshold).astype(int)
    f1 = f1_score(labels, predictions)
    f1_scores.append(f1)

# Find best threshold
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# Calculate metrics at best threshold
best_predictions = (scores >= best_threshold).astype(int)
tp = np.sum((labels == 1) & (best_predictions == 1))
fp = np.sum((labels == 0) & (best_predictions == 1))
fn = np.sum((labels == 1) & (best_predictions == 0))
tn = np.sum((labels == 0) & (best_predictions == 0))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
accuracy = (tp + tn) / len(labels)

print(f"\n{'='*60}")
print("OPTIMAL THRESHOLD ANALYSIS")
print('='*60)
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")
print(f"\nMetrics at best threshold:")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TP: {tp:6d}  FP: {fp:6d}")
print(f"  FN: {fn:6d}  TN: {tn:6d}")
print('='*60)

# ---------------- PLOT ROC + PR ----------------
fpr, tpr, roc_thresholds = roc_curve(labels, scores)
prec, rec, pr_thresholds = precision_recall_curve(labels, scores)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
plt.scatter([fp/(fp+tn)], [tp/(tp+fn)], color='red', s=100, zorder=5, 
            label=f'Best threshold = {best_threshold:.4f}')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(rec, prec, label=f"AUC = {auc(rec, prec):.4f}")
plt.scatter([recall], [precision], color='red', s=100, zorder=5,
            label=f'Best threshold = {best_threshold:.4f}')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.show()

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"AUC ROC: {auc(fpr, tpr):.4f}")
print(f"AUC PR:  {auc(rec, prec):.4f}")
print(f"Best Threshold: {best_threshold:.4f}")
print(f"Best F1 Score:  {best_f1:.4f}")
print('='*60)
