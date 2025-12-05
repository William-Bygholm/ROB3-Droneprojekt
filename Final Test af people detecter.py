import cv2
import joblib
import numpy as np
import json
import time

# ---------------- USER CONFIG ----------------
VIDEOS = [
    r"ProjektVideoer/2 mili en idiot der ligger ned.MP4",
    r"ProjektVideoer/3 mili 2 onde 1 god.MP4"
]
JSON_FILES = [
    r"Testing/2 mili og 1 idiot.json",
    r"Testing/3mili 2 onde 1 god.json"
]
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"
BEST_THRESHOLD = 1  # indsæt dit valgte threshold her

SCALES = [1.25, 1, 0.8, 0.64]
STEP_SIZES = {1.25: 24, 1: 24, 0.8: 20, 0.64: 16}
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

# ---------------- FINAL TEST ----------------
tp_total, fp_total, fn_total = 0, 0, 0
start_time = time.time()
total_frames = 0

# Beregn antal frames til estimeret tid (med FRAME_SKIP)
for vid in VIDEOS:
    cap = cv2.VideoCapture(vid)
    total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // FRAME_SKIP
    cap.release()

processed_frames = 0

for vid, json_file in zip(VIDEOS, JSON_FILES):
    with open(json_file, "r") as f:
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

    cap = cv2.VideoCapture(vid)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        # FRAME_SKIP logik: spring frames over
        if (frame_id - 1) % FRAME_SKIP != 0:
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

        # ---------------- TP/FP/FN ----------------
        matched_gt = [False]*len(gt_boxes_scaled)
        matched_det = [False]*len(final_boxes)

        # Match TP
        for i, gt in enumerate(gt_boxes_scaled):
            for j, (box, score) in enumerate(zip(final_boxes, final_scores)):
                if score >= BEST_THRESHOLD and iou(box, gt) > IOU_POSITIVE and not matched_det[j]:
                    matched_gt[i] = True
                    matched_det[j] = True
                    break

        # TP/FN
        for m in matched_gt:
            if m:
                tp_total += 1
            else:
                fn_total += 1

        # FP
        for j, score in enumerate(final_scores):
            if score >= BEST_THRESHOLD and not matched_det[j]:
                fp_total += 1

        # Print tid + TP/FP/FN
        processed_frames += 1
        elapsed = time.time() - start_time
        progress = processed_frames / total_frames
        if progress > 0:
            est_total = elapsed / progress
            remaining = est_total - elapsed
            print(f"Processed frame {processed_frames}/{total_frames} "
                  f"({progress*100:.2f}%), Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s | "
                  f"TP: {tp_total}, FP: {fp_total}, FN: {fn_total}", end="\r")

    cap.release()

# ---------------- METRICS ----------------
precision = tp_total / (tp_total+fp_total) if (tp_total+fp_total) > 0 else 0
recall = tp_total / (tp_total+fn_total) if (tp_total+fn_total) > 0 else 0
f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

print(f"\nFINAL TEST RESULTS:")
print(f"TP: {tp_total}, FP: {fp_total}, FN: {fn_total}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
