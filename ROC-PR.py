import cv2
import joblib
import numpy as np
import json
import time
import re
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score

# ---------------- USER CONFIG ----------------
VIDEO_IN = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4"
JSON_COCO = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\Validation\2 mili med blå bond.json"
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"

SCALES = [1.2, 1, 0.8, 0.64]
STEP_SIZES = {1.2: 36, 1: 34, 0.8: 28, 0.64: 24}
NMS_THRESHOLD = 0.05
FRAME_SKIP = 200
WINDOW_SIZE = (128, 256)
IOU_POSITIVE = 0.3

# Debug / output
SAVE_PER_FRAME_CSV = True
PER_FRAME_CSV = "per_frame_stats.csv"

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model_data = joblib.load(MODEL_FILE)
if isinstance(model_data, dict):
    clf = model_data.get("classifier", model_data)
    hog_params = model_data.get("hog_params", None)
    if hog_params is not None:
        WINDOW_SIZE = hog_params.get("winSize", WINDOW_SIZE)
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
    area_a = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    area_b = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    return inter / (area_a + area_b - inter + 1e-9)

def extract_frame_number_from_filename(name):
    """
    Try to extract an integer frame index from a filename like 'frame_000123.png' or '000123.jpg'.
    Returns None if no digits found.
    """
    digits = re.findall(r'\d+', name)
    if not digits:
        return None
    # choose longest group of digits (likely the frame number)
    candidate = max(digits, key=len)
    try:
        return int(candidate)
    except:
        return None

# ---------------- LOAD COCO JSON & BUILD MAPPING ----------------
with open(JSON_COCO, "r") as f:
    coco = json.load(f)

# Try robust image -> frame mapping:
image_id_to_frame = {}
# If file names have frame numbers, use them. Otherwise fallback to enumerate (1-based).
for idx, img in enumerate(coco.get("images", [])):
    img_id = img.get("id")
    fname = img.get("file_name", "")
    frame_num = extract_frame_number_from_filename(fname)
    if frame_num is None:
        # fallback to 1-based index (be cautious; may be wrong for some exports)
        frame_num = idx + 1
    image_id_to_frame[img_id] = frame_num

frame_to_boxes = {}
for ann in coco.get("annotations", []):
    frame_id = image_id_to_frame.get(ann["image_id"])
    if frame_id is None:
        continue
    x, y, w, h = ann["bbox"]
    frame_to_boxes.setdefault(frame_id, []).append([int(x), int(y), int(x+w), int(y+h)])

# If COCO image size available:
first_image = coco.get("images", [{}])[0]
coco_w, coco_h = first_image.get("width", None), first_image.get("height", None)
if coco_w is None or coco_h is None:
    print("[WARN] COCO image width/height not found. Scaling may be incorrect if video and JSON differ.")

# ---------------- VALIDATION LOOP ----------------
det_scores = []   # scores for detections only
det_labels = []   # 1 if detection matched a gt, else 0
total_gts = 0     # total number of ground-truth persons across frames

# Optional record per-frame stats
if SAVE_PER_FRAME_CSV:
    csv_file = open(PER_FRAME_CSV, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "num_gt", "num_detections", "num_matched_gt", "num_unmatched_gt", "num_tp_dets", "num_fp_dets"])

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
    # If coco dimensions known, scale; else assume equal
    if coco_w and coco_h:
        scale_x = w0 / coco_w
        scale_y = h0 / coco_h
    else:
        scale_x = scale_y = 1.0

    # Load GT boxes for current frame (based on mapping above)
    gt_boxes = frame_to_boxes.get(frame_id, [])
    gt_boxes_scaled = [[int(x1*scale_x), int(y1*scale_y),
                        int(x2*scale_x), int(y2*scale_y)]
                        for x1,y1,x2,y2 in gt_boxes]

    total_gts += len(gt_boxes_scaled)

    detections, scores = [], []

    # ------ Run detector ------
    for scale in SCALES:
        resized = cv2.resize(gray, None, fx=scale, fy=scale)
        step = STEP_SIZES.get(scale, max(8, int(32*scale//1)))
        scale_x_win = w0 / float(resized.shape[1])
        scale_y_win = h0 / float(resized.shape[0])

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
                scores.append(float(score))

    # NMS + optional merge
    nms_boxes, nms_scores = nms_opencv(detections, scores, 0.0, NMS_THRESHOLD)
    final_boxes, final_scores = merge_close_boxes(nms_boxes, nms_scores)

    # ------ MATCH DETECTIONS TO GT (correct) ------
    gt_used = [False] * len(gt_boxes_scaled)

    # First, for stable matching we find best-match per detection (greedy by best IoU)
    # We'll perform a greedy match: iterate detections sorted by score desc, match to best unused GT
    order = np.argsort(final_scores)[::-1] if final_scores else []
    num_tp_dets = 0
    num_fp_dets = 0

    for idx in order:
        det_box = final_boxes[idx]
        det_score = final_scores[idx]
        best_i = -1
        best_iou = 0.0
        for i, g in enumerate(gt_boxes_scaled):
            if gt_used[i]:
                continue
            val = iou(det_box, g)
            if val > best_iou:
                best_iou = val
                best_i = i
        if best_i >= 0 and best_iou > IOU_POSITIVE:
            # match
            gt_used[best_i] = True
            det_labels.append(1)
            num_tp_dets += 1
        else:
            det_labels.append(0)
            num_fp_dets += 1
        det_scores.append(final_scores[idx])

    # If there were no detections, nothing appended to det_scores/labels for this frame
    # Count unmatched GTs (these are FNs)
    num_unmatched_gt = sum(1 for used in gt_used if not used)
    num_matched_gt = len(gt_boxes_scaled) - num_unmatched_gt

    # Write per-frame stats
    if SAVE_PER_FRAME_CSV:
        csv_writer.writerow([frame_id, len(gt_boxes_scaled), len(final_boxes),
                             num_matched_gt, num_unmatched_gt, num_tp_dets, num_fp_dets])

    # Progress print
    elapsed = time.time() - start_time
    # Use number of frames read as progress baseline
    progress = frame_id / float(total_frames) if total_frames > 0 else 0
    if progress > 0:
        est_total = elapsed / progress
        remaining = est_total - elapsed
        print(f"Frame {frame_id}/{total_frames} ({progress*100:.2f}%), "
              f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s", end="\r")

cap.release()
if SAVE_PER_FRAME_CSV:
    csv_file.close()

print(f"\n[INFO] Validation complete. Collected {len(det_scores)} detection samples. Total GTs: {total_gts}")

# ---------------- METRICS ----------------
det_scores = np.array(det_scores, dtype=np.float32)
det_labels = np.array(det_labels, dtype=np.int32)

if len(det_scores) == 0:
    raise RuntimeError("No detections collected. Check detector or frame skipping.")

# Find best threshold using detection-only samples
thresholds = np.linspace(det_scores.min(), det_scores.max(), 200)
f1_scores = [f1_score(det_labels, (det_scores >= thr).astype(int)) for thr in thresholds]
best_idx = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])
best_f1 = float(f1_scores[best_idx])

# Predictions on detection samples at chosen threshold
det_preds = (det_scores >= best_threshold).astype(int)
tp_dets = int(np.sum((det_labels == 1) & (det_preds == 1)))
fp_dets = int(np.sum((det_labels == 0) & (det_preds == 1)))

# Correct FN counting: people missed entirely = total_gts - TP_from_detections
fn_total = int(total_gts - tp_dets)
if fn_total < 0:
    # safety clamp (shouldn't happen)
    fn_total = 0

# TN is not well-defined in sliding-window detection (infinite negatives). We'll print 0 for clarity.
tn_total = 0

precision = tp_dets / (tp_dets + fp_dets) if (tp_dets + fp_dets) > 0 else 0.0
recall = tp_dets / (tp_dets + fn_total) if (tp_dets + fn_total) > 0 else 0.0
accuracy = tp_dets / (tp_dets + fp_dets + fn_total) if (tp_dets + fp_dets + fn_total) > 0 else 0.0

print(f"\nBest Threshold: {best_threshold:.4f} | F1 (det-only): {best_f1:.4f} | "
      f"Precision: {precision:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f}")
print(f"TP: {tp_dets}, FP: {fp_dets}, FN: {fn_total}")

# ---------------- PLOT (based on detection samples) ----------------
fpr, tpr, _ = roc_curve(det_labels, det_scores)
prec, rec, _ = precision_recall_curve(det_labels, det_scores)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
plt.scatter([fp_dets/(fp_dets+tn_total+1e-9)], [tp_dets/(tp_dets+fn_total+1e-9)], color='red', s=80,
            label=f'Best thr = {best_threshold:.4f}')
plt.title("ROC Curve (detection-level)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(rec, prec, label=f"Precision/Recall")
plt.scatter([recall], [precision], color='red', s=80, label=f'Best thr = {best_threshold:.4f}')
plt.title("Precision-Recall Curve (detection-level)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.show()
