import cv2
import re
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score
import csv
import os
import time

# ---------------- USER CONFIG ----------------
TEST_VIDEOS = [
    {"video": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 mili en idiot der ligger ned.MP4", 
     "coco": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json"},
    {"video": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\3 mili 2 onde 1 god.MP4", 
     "coco": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\3mili 2 onde 1 god.json"}
]
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"
THRESHOLD = 1.0590
WINDOW_SIZE = (128, 256)
STEP_SIZE = 48
IOU_POSITIVE = 0.5
FRAME_SKIP = 2  # spring hver 2. frame
OUTPUT_CSV = "combined_test_results.csv"
PLOTS_DIR = "test_plots_excel_1"    

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model_data = joblib.load(MODEL_FILE)
clf = model_data.get("classifier", model_data) if isinstance(model_data, dict) else model_data
hog = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(32,32),
    _blockStride=(16,16),
    _cellSize=(8,8),
    _nbins=9
)

# --- NMS & Box Merging (kopieret fra SVM_In_Action.py) ---
def nms_opencv(detections, scores, score_threshold, nms_threshold):
    if len(detections) == 0:
        return []
    boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in detections]
    scores = [float(s) for s in scores]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold, nms_threshold)
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    return [detections[i] for i in indices]

def merge_close_boxes(boxes, iou_threshold=0.2):
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        group = [boxes[i]]
        used[i] = True
        for j in range(i+1, len(boxes)):
            if used[j]:
                continue
            xx1, yy1, xx2, yy2 = boxes[j]
            inter_x1 = max(x1, xx1)
            inter_y1 = max(y1, yy1)
            inter_x2 = min(x2, xx2)
            inter_y2 = min(y2, yy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xx2 - xx1) * (yy2 - yy1)
            iou = inter_area / float(area1 + area2 - inter_area + 1e-9)
            if iou > iou_threshold:
                group.append(boxes[j])
                used[j] = True
        gx1 = min(b[0] for b in group)
        gy1 = min(b[1] for b in group)
        gx2 = max(b[2] for b in group)
        gy2 = max(b[3] for b in group)
        merged.append([gx1, gy1, gx2, gy2])
    return merged

NMS_THRESHOLD = 0.05
MERGE_IOU_THRESHOLD = 0.2
SVM_THRESHOLD = THRESHOLD
def sliding_windows(img, step, win_size):
    w, h = win_size
    for y in range(0, img.shape[0]-h+1, step):
        for x in range(0, img.shape[1]-w+1, step):
            yield x, y, img[y:y+h, x:x+w]

def iou(a, b):
    x1, y1, x2, y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

# ---------------- RUN TEST FOR ALL VIDEOS ----------------
scores_all, labels_all = [], []

total_frames_all = 0
for test in TEST_VIDEOS:
    cap_tmp = cv2.VideoCapture(test["video"])
    total_frames_all += int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) // FRAME_SKIP
    cap_tmp.release()

processed_frames = 0
start_time_total = time.time()

# totals across all videos (from greedy matching)
TP_total = 0
FP_total = 0
FN_total = 0

for test in TEST_VIDEOS:
    video_path = test["video"]
    coco_path = test["coco"]

    with open(coco_path) as f:
        coco = json.load(f)
    # Map COCO image ids to video frame indices
    # Extract numeric index from file_name (e.g. 'frame_000123.png' -> 123)
    image_id_to_frame = {}
    for img in coco.get("images", []):
        img_id = img.get("id")
        fname = img.get("file_name", "")
        m = re.search(r"(\d+)", fname)
        frame_idx = int(m.group(1)) if m else 0
        image_id_to_frame[img_id] = frame_idx

    frame_to_boxes = {}
    for ann in coco.get("annotations", []):
        frame_id = image_id_to_frame.get(ann["image_id"], ann["image_id"])
        x, y, w, h = ann["bbox"]
        frame_to_boxes.setdefault(frame_id, []).append([int(x), int(y), int(x+w), int(y+h)])
    coco_w = coco.get("images", [])[0].get("width") if coco.get("images") else None
    coco_h = coco.get("images", [])[0].get("height") if coco.get("images") else None

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time_video = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        if frame_id % FRAME_SKIP != 0:
            continue

        processed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray.shape[:2]
        # if coco sizes present, scale GT to video frame size
        if coco_w and coco_h:
            scale_x = w0 / coco_w
            scale_y = h0 / coco_h
        else:
            scale_x = scale_y = 1.0

        gt_boxes = frame_to_boxes.get(frame_id, [])
        gt_boxes_scaled = [[int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)] for x1,y1,x2,y2 in gt_boxes]

        detections, det_scores = [], []
        for x, y, win in sliding_windows(gray, STEP_SIZE, WINDOW_SIZE):
            feat = hog.compute(win).ravel()
            score = clf.decision_function([feat])[0]
            detections.append([x, y, x+WINDOW_SIZE[0], y+WINDOW_SIZE[1]])
            det_scores.append(score)

        # --- NMS + Box Merging to match SVM_In_Action pipeline ---
        nms_boxes = nms_opencv(detections, det_scores, SVM_THRESHOLD, NMS_THRESHOLD)
        final_boxes = merge_close_boxes(nms_boxes, iou_threshold=MERGE_IOU_THRESHOLD)

        # Perform greedy matching between final_boxes and GTs by IoU
        matched_gt = set()
        matched_det = set()
        # build IoU matrix
        iou_matrix = []
        for i, d in enumerate(final_boxes):
            row = []
            for j, g in enumerate(gt_boxes_scaled):
                row.append(iou(d, g))
            iou_matrix.append(row)

        # greedy match highest IoU pairs
        pairs = []
        for i, row in enumerate(iou_matrix):
            for j, val in enumerate(row):
                pairs.append((val, i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        for val, i, j in pairs:
            if val <= IOU_POSITIVE:
                break
            if i in matched_det or j in matched_gt:
                continue
            matched_det.add(i)
            matched_gt.add(j)

        # Count TP/FP/FN for this frame
        frame_TP = len(matched_det)
        frame_FP = len(final_boxes) - len(matched_det)
        frame_FN = len(gt_boxes_scaled) - len(matched_gt)

        # append detection scores/labels for PR curve: matched detections -> label 1 else 0
        for i, box in enumerate(final_boxes):
            lbl = 1 if i in matched_det else 0
            # Score for merged box = max score of original detections that overlap it
            group_scores = []
            for j, det in enumerate(detections):
                if iou(box, det) > 0.5:
                    group_scores.append(det_scores[j])
            score = max(group_scores) if group_scores else 0.0
            scores_all.append(score)
            labels_all.append(lbl)

        # accumulate counts
        TP_total += frame_TP
        FP_total += frame_FP
        FN_total += frame_FN

        # --- Progress print ---
        if frame_id % 50 == 0 or frame_id == total_frames:
            elapsed_total = time.time() - start_time_total
            progress_total = processed_frames / total_frames_all if total_frames_all > 0 else 0
            est_total_time = elapsed_total / progress_total if progress_total > 0 else 0
            remaining_time = est_total_time - elapsed_total
            bar_length = 30
            filled = int(bar_length * progress_total)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"[{bar}] {progress_total*100:5.1f}% | {processed_frames}/{total_frames_all} frames | "
                  f"Elapsed: {int(elapsed_total):3d}s | Est. remaining: {int(remaining_time):3d}s", end="\r")

    cap.release()
    print()

# ---------------- CALCULATE METRICS ----------------
# Add missed GTs (FN_total) to PR arrays with a very low score so they are counted as positives
if FN_total > 0:
    if len(scores_all) > 0:
        low_score = min(scores_all) - 1.0
    else:
        low_score = THRESHOLD - 1.0
    for _ in range(FN_total):
        scores_all.append(low_score)
        labels_all.append(1)

labels_all = np.array(labels_all)
scores_all = np.array(scores_all)

# preds for threshold-based classification (used only for f1 if desired)
preds = (scores_all >= THRESHOLD).astype(int)

# Use the matched counts we computed per-frame for final TP/FP/FN
TP = int(TP_total)
FP = int(FP_total)
FN = int(FN_total)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

# PR curve and AUC from all scored samples
prec, rec, _ = precision_recall_curve(labels_all, scores_all)
pr_auc = auc(rec, prec)

# ---------------- PRINT METRICS ----------------
print("\n[RESULTS] Combined Test Metrics (TN ignored):")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"PR AUC: {pr_auc:.4f}")

# ---------------- SAVE CSV ----------------
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["TP", TP])
    writer.writerow(["FP", FP])
    writer.writerow(["FN", FN])
    writer.writerow(["Precision", precision])
    writer.writerow(["Recall", recall])
    writer.writerow(["F1", f1])
    writer.writerow(["PR AUC", pr_auc])
    writer.writerow([])
    writer.writerow(["PR Recall"] + rec.tolist())
    writer.writerow(["PR Precision"] + prec.tolist())

print(f"[INFO] CSV saved to {OUTPUT_CSV}")

# ---------------- SAVE GRAPHS ----------------
plt.figure()
plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.4f}")
plt.scatter([recall], [precision], color='red', label=f'Threshold={THRESHOLD}')
plt.title("Precision-Recall Curve - Combined Videos")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "Combined_PR.png"))
plt.close()

print(f"[INFO] Graphs saved to {PLOTS_DIR}")
