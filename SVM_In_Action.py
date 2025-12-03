# detector.py
import cv2
import joblib
import numpy as np

VIDEO_IN = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4"
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"

SCALES = [1.0, 0.8]
STEP_SIZES = {1.0: 32, 0.8: 28}

NMS_THRESHOLD = 0.05
DISPLAY_SCALE = 0.3
FRAME_SKIP = 20
SVM_THRESHOLD = 0.8


# ---------------- LOAD MODEL ----------------
print("Loading model...")
model_data = joblib.load(MODEL_FILE)

# Fallback hvis pickle kun indeholder SVM
if isinstance(model_data, dict):
    clf = model_data.get("classifier", model_data)
    hog_params = model_data.get("hog_params", None)
    if hog_params is not None:
        WINDOW_SIZE = hog_params.get("winSize", (128, 256))
        print(f"Loaded model with custom HOG params: {WINDOW_SIZE}")
    else:
        WINDOW_SIZE = (128, 256)
        print("No hog_params in model dict, using default HOG window size.")
else:
    clf = model_data
    WINDOW_SIZE = (128, 256)
    print("Loaded SVM directly, using default HOG window size.")

# HOG descriptor
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
    for y in range(0, img.shape[0] - h + 1, step):
        for x in range(0, img.shape[1] - w + 1, step):
            yield x, y, img[y:y+h, x:x+w]


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


# ---------------- DETECTION FUNCTION ----------------
def detect_people(frame, clf, hog):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape[:2]

    detections = []
    scores = []

    for scale in SCALES:
        resized = cv2.resize(gray, None, fx=scale, fy=scale)
        step = STEP_SIZES[scale]

        scale_x = w0 / resized.shape[1]
        scale_y = h0 / resized.shape[0]

        for x, y, win in sliding_windows(resized, step, WINDOW_SIZE):
            if win.shape != (WINDOW_SIZE[1], WINDOW_SIZE[0]):
                continue

            feat = hog.compute(win).ravel()
            score = clf.decision_function([feat])[0]

            if score > SVM_THRESHOLD:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + WINDOW_SIZE[0]) * scale_x)
                y2 = int((y + WINDOW_SIZE[1]) * scale_y)
                detections.append([x1, y1, x2, y2])
                scores.append(score)

    # NMS + merge
    nms_boxes = nms_opencv(detections, scores, SVM_THRESHOLD, NMS_THRESHOLD)
    final_boxes = merge_close_boxes(nms_boxes)

    return final_boxes


# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_IN)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        boxes = detect_people(frame, clf, hog)

        # Draw boxes
        out = frame.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

        disp = cv2.resize(out, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        cv2.imshow("HOG+SVM Detector", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
