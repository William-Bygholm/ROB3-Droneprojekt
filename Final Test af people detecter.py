import cv2
import joblib
import numpy as np
import json

# ---------------- CONFIG ----------------
VIDEO_IN = r"ProjektVideoer/2 militær med blå bånd .MP4"
JSON_COCO = r"Validation/2 mili med blå bond.json"
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"

SCALES = [1.25, 1, 0.8, 0.64]
STEP_SIZES = {1.25: 24, 1: 24, 0.8: 20, 0.64: 16}
NMS_THRESHOLD = 0.1
FRAME_SKIP = 100
WINDOW_SIZE = (128, 256)
IOU_POSITIVE = 0.5

# ---------------- LOAD MODEL ----------------
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
    _blockSize=(32,32),
    _blockStride=(16,16),
    _cellSize=(8,8),
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

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
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

# ---------------- VIDEO LOOP ----------------
cap = cv2.VideoCapture(VIDEO_IN)
frame_id = 0

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

    # ---------------- DETECTION ----------------
    detections, scores = [], []
    for scale in SCALES:
        resized = cv2.resize(gray, None, fx=scale, fy=scale)
        step = STEP_SIZES[scale]
        scale_x_win = w0 / resized.shape[1]
        scale_y_win = h0 / resized.shape[0]

        for x, y, win in sliding_windows(resized, step, WINDOW_SIZE):
            feat = hog.compute(win).ravel().reshape(1, -1)
            score = clf.decision_function(feat)[0]
            x1 = int(x * scale_x_win)
            y1 = int(y * scale_y_win)
            x2 = int((x + WINDOW_SIZE[0]) * scale_x_win)
            y2 = int((y + WINDOW_SIZE[1]) * scale_y_win)
            detections.append([x1, y1, x2, y2])
            scores.append(score)

    nms_boxes, nms_scores = nms_opencv(detections, scores, 0.0, NMS_THRESHOLD)

    # ---------------- CLASSIFY TP/FP/FN ----------------
    matched_gt = set()
    for det in nms_boxes:
        match = False
        for i, gt in enumerate(gt_boxes_scaled):
            if i in matched_gt:
                continue
            if iou(det, gt) > IOU_POSITIVE:
                match = True
                matched_gt.add(i)
                break
        color = (0,255,0) if match else (0,0,255)  # Grøn=TP, Rød=FP
        cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), color, 2)

    # Tegn FN (gt ikke matchet)
    for i, gt in enumerate(gt_boxes_scaled):
        if i not in matched_gt:
            cv2.rectangle(frame, (gt[0], gt[1]), (gt[2], gt[3]), (255,0,0), 2)  # Blå=FN

    # Tegn alle GT (valgfrit)
    for gt in gt_boxes_scaled:
        cv2.rectangle(frame, (gt[0], gt[1]), (gt[2], gt[3]), (255,255,255), 1)  # Hvid=GT ramme

    cv2.imshow("Detections", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC for at lukke
        break

cap.release()
cv2.destroyAllWindows()
