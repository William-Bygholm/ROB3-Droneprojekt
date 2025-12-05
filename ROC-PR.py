import cv2
import joblib
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score

# ---------------- USER CONFIG ----------------
VIDEO_IN = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4"
JSON_COCO = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\Validation\2 mili med blå bond.json"
MODEL_FILE = "Person_Detector_Json.pkl"

SCALES = [1.2, 1, 0.8, 0.64]
STEP_SIZES = {1.2: 36, 1: 34, 0.8: 28, 0.64: 24}
NMS_THRESHOLD = 0.05
FRAME_SKIP = 2
WINDOW_SIZE = (128, 256)
IOU_POSITIVE = 0.5

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model_data = joblib.load(MODEL_FILE)
if isinstance(model_data, dict):
    clf = model_data.get("classifier", model_data)
    hog_params = model_data.get("hog_params", None)
    if hog_params is not None:
        WINDOW_SIZE = hog_params.get("winSize", (128, 256))
        print(f"Loaded model with custom HOG params: {WINDOW_SIZE}")
else:
    clf = model_data
    print("Loaded SVM directly.")

hog = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(8, 8),
    _nbins=9
)

# ---------------- HELPERS ----------------
def sliding_windows(img, step, win_size):
    w, h = win_size
    for y in range(0, img.shape[0]-h+1, step):
        for x in range(0, img.shape[1]-w+1, step):
            yield x, y, img[y:y+h, x:x+w]

def nms_opencv(detections, scores, score_thresh, nms_thresh):
    if len(detections) == 0:
        return [], []
    boxes_xywh = np.array([[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2 in detections], dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), score_thresh, nms_thresh)
    if len(indices) == 0:
        return [], []
    indices = indices.flatten()
    return [detections[i] for i in indices], [scores[i] for i in indices]

def merge_close_boxes(boxes, scores, iou_threshold=0.2):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    keep = []
    while len(boxes) > 0:
        box = boxes[0]
        score = scores[0]
        keep.append((box.tolist(), score))
        boxes = boxes[1:]
        scores = scores[1:]
        if len(boxes) == 0:
            break
        xx1 = np.maximum(box[0], boxes[:,0])
        yy1 = np.maximum(box[1], boxes[:,1])
        xx2 = np.minimum(box[2], boxes[:,2])
        yy2 = np.minimum(box[3], boxes[:,3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area1 = (box[2]-box[0])*(box[3]-box[1])
        area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        iou_vals = inter / (area1 + area2 - inter + 1e-9)
        mask = iou_vals <= iou_threshold
        boxes = boxes[mask]
        scores = scores[mask]
    if keep:
        final_boxes, final_scores = zip(*keep)
    else:
        final_boxes, final_scores = [], []
    return list(final_boxes), list(final_scores)

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

# ---------------- LOAD COCO JSON ----------------
with open(JSON_COCO, "r") as f:
    coco = json.load(f)

image_id_to_frame = {img["id"]: idx+1 for idx, img in enumerate(coco["images"])}
frame_to_boxes = {}
for ann in coco["annotations"]:
    frame_id = image_id_to_frame.get(ann["image_id"])
    if frame_id is None:
        continue
    x, y, w, h = ann["bbox"]
    frame_to_boxes.setdefault(frame_id, []).append([int(x), int(y), int(x+w), int(y+h)])

coco_w, coco_h = coco["images"][0]["width"], coco["images"][0]["height"]

# ---------------- VALIDATION LOOP ----------------
scores_all, labels_all = [], []
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
    scale_x = w0 / coco_w
    scale_y = h0 / coco_h
    gt_boxes = frame_to_boxes.get(frame_id, [])
    gt_boxes_scaled = [[int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)]
                       for x1,y1,x2,y2 in gt_boxes]

    detections, scores = [], []

    for scale in SCALES:
        resized = cv2.resize(gray, None, fx=scale, fy=scale)
        step = STEP_SIZES[scale]
        scale_x_win = w0 / resized.shape[1]
        scale_y_win = h0 / resized.shape[0]
        windows = list(sliding_windows(resized, step, WINDOW_SIZE))
        feats = [hog.compute(win).ravel() for _,_,win in windows]
        if feats:
            scores_batch = clf.decision_function(feats)
            for (x, y, _), score in zip(windows, scores_batch):
                x1 = int(x * scale_x_win)
                y1 = int(y * scale_y_win)
                x2 = int((x + WINDOW_SIZE[0]) * scale_x_win)
                y2 = int((y + WINDOW_SIZE[1]) * scale_y_win)
                detections.append([x1, y1, x2, y2])
                scores.append(score)

    nms_boxes, nms_scores = nms_opencv(detections, scores, 0.0, NMS_THRESHOLD)
    final_boxes, final_scores = merge_close_boxes(nms_boxes, nms_scores)

    for box, score in zip(final_boxes, final_scores):
        label = 0
        for g in gt_boxes_scaled:
            if iou(box, g) > IOU_POSITIVE:
                label = 1
                break
        scores_all.append(score)
        labels_all.append(label)

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
print(f"\n[INFO] Validation complete. Collected {len(scores_all)} detections.")

# ---------------- METRICS ----------------
scores = np.array(scores_all, dtype=np.float32)
labels = np.array(labels_all, dtype=np.int32)

thresholds = np.linspace(scores.min(), scores.max(), 100)
f1_scores = [f1_score(labels, (scores >= thr).astype(int)) for thr in thresholds]
best_idx = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])
best_f1 = float(f1_scores[best_idx])

best_predictions = (scores >= best_threshold).astype(int)
tp = int(np.sum((labels == 1) & (best_predictions == 1)))
fp = int(np.sum((labels == 0) & (best_predictions == 1)))
fn = int(np.sum((labels == 1) & (best_predictions == 0)))
tn = int(np.sum((labels == 0) & (best_predictions == 0)))

precision = tp / (tp+fp) if (tp+fp) > 0 else 0
recall = tp / (tp+fn) if (tp+fn) > 0 else 0
accuracy = (tp+tn)/len(labels)

print(f"\nBest Threshold: {best_threshold:.4f} | F1: {best_f1:.4f} | "
      f"Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

# ---------------- PLOT ----------------
fpr, tpr, _ = roc_curve(labels, scores)
prec, rec, _ = precision_recall_curve(labels, scores)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
plt.scatter([fp/(fp+tn+1e-9)], [tp/(tp+fn+1e-9)], color='red', s=100,
            label=f'Best thr = {best_threshold:.4f}')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(rec, prec, label=f"AUC = {auc(rec, prec):.4f}")
plt.scatter([recall], [precision], color='red', s=100, label=f'Best thr = {best_threshold:.4f}')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.show()