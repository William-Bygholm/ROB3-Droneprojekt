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

JSONS = [
    r"Testing/2 mili og 1 idiot.json",
    r"Testing/3mili 2 onde 1 god.json"
]

MODEL_FILE = "Person_Detector_Json+YOLO.pkl"

SCALES = [1.2, 1, 0.8, 0.64]
STEP_SIZES = {1.2: 36, 1: 34, 0.8: 28, 0.64: 24}
NMS_THRESHOLD = 0.05
FRAME_SKIP = 2
WINDOW_SIZE = (128, 256)
IOU_POSITIVE = 0.3

# ✅ FAST THRESHOLD
THRESHOLD = x


# ---------------- LOAD MODEL ----------------
print("Loading model...")
model_data = joblib.load(MODEL_FILE)
if isinstance(model_data, dict):
    clf = model_data.get("classifier", model_data)
    hog_params = model_data.get("hog_params", None)
    if hog_params is not None:
        WINDOW_SIZE = hog_params.get("winSize", (128, 256))
else:
    clf = model_data

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
    idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), score_thresh, nms_thresh)
    if len(idx) == 0:
        return [], []
    idx = idx.flatten()
    return [detections[i] for i in idx], [scores[i] for i in idx]


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
        b, s = zip(*keep)
        return list(b), list(s)
    return [], []


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)


# ---------------- PROCESS ONE VIDEO+JSON ----------------
def process_pair(video_path, json_path):
    print(f"\nProcessing: {video_path}")

    with open(json_path, "r") as f:
        coco = json.load(f)

    image_id_to_frame = {img["id"]: idx+1 for idx, img in enumerate(coco["images"])}
    frame_to_boxes = {}
    for ann in coco["annotations"]:
        fid = image_id_to_frame.get(ann["image_id"])
        if fid is None:
            continue
        x, y, w, h = ann["bbox"]
        frame_to_boxes.setdefault(fid, []).append([int(x), int(y), int(x+w), int(y+h)])

    coco_w, coco_h = coco["images"][0]["width"], coco["images"][0]["height"]

    scores_all, labels_all = [], []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0

    start_time = time.time()

    running_tp = 0
    running_fp = 0
    running_fn = 0

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
        gt_scaled = [[int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)]
                     for x1,y1,x2,y2 in gt_boxes]

        detections, scores = [], []

        for scale in SCALES:
            resized = cv2.resize(gray, None, fx=scale, fy=scale)
            step = STEP_SIZES[scale]
            sx = w0 / resized.shape[1]
            sy = h0 / resized.shape[0]

            windows = list(sliding_windows(resized, step, WINDOW_SIZE))
            feats = [hog.compute(win).ravel() for _,_,win in windows]

            if feats:
                scores_batch = clf.decision_function(feats)
                for (x, y, _), sc in zip(windows, scores_batch):
                    x1 = int(x * sx)
                    y1 = int(y * sy)
                    x2 = int((x + WINDOW_SIZE[0]) * sx)
                    y2 = int((y + WINDOW_SIZE[1]) * sy)
                    detections.append([x1, y1, x2, y2])
                    scores.append(sc)

        nms_boxes, nms_scores = nms_opencv(detections, scores, 0.0, NMS_THRESHOLD)
        final_boxes, final_scores = merge_close_boxes(nms_boxes, nms_scores)

        frame_tp = 0
        frame_fp = 0
        frame_fn = 0

        matched_gt = set()

        for box, sc in zip(final_boxes, final_scores):
            is_tp = False
            for i, g in enumerate(gt_scaled):
                if i in matched_gt:
                    continue
                if iou(box, g) > IOU_POSITIVE:
                    is_tp = True
                    matched_gt.add(i)
                    break

            if sc >= THRESHOLD:
                if is_tp:
                    frame_tp += 1
                else:
                    frame_fp += 1

        frame_fn = len(gt_scaled) - len(matched_gt)

        running_tp += frame_tp
        running_fp += frame_fp
        running_fn += frame_fn

        elapsed = time.time() - start_time
        progress = frame_id / total_frames
        remaining = (elapsed / progress) - elapsed if progress > 0 else 0

        print(
            f"Frame {frame_id}/{total_frames} | "
            f"TP: {running_tp}  FP: {running_fp}  FN: {running_fn} | "
            f"Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s",
            end="\r"
        )

        for box, sc in zip(final_boxes, final_scores):
            label = 0
            for g in gt_scaled:
                if iou(box, g) > IOU_POSITIVE:
                    label = 1
                    break
            scores_all.append(sc)
            labels_all.append(label)

    cap.release()
    print()
    return scores_all, labels_all


# ---------------- RUN BOTH VIDEOS ----------------
all_scores = []
all_labels = []

for vid, js in zip(VIDEOS, JSONS):
    s, l = process_pair(vid, js)
    all_scores.extend(s)
    all_labels.extend(l)

all_scores = np.array(all_scores, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int32)

# ---------------- METRICS ----------------
pred = (all_scores >= THRESHOLD).astype(int)

tp = int(np.sum((all_labels == 1) & (pred == 1)))
fp = int(np.sum((all_labels == 0) & (pred == 1)))
fn = int(np.sum((all_labels == 1) & (pred == 0)))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("\n===== FINAL METRICS (2 videoer samlet) =====")
print(f"Threshold: {THRESHOLD}")
print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
