import cv2
import numpy as np

# Load cascades
cascade_body = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
cascade_upper = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# Open video
cap = cv2.VideoCapture("C:/Users/alexa/Downloads/Civil person.MP4")
resize_scale = 0.5

# For stabilization
prev_gray = None

# For temporal filtering
detection_history = []   # list of boxes from past frames
HISTORY_SIZE = 7         # smoothing window

def merge_boxes(boxes):
    """Merges overlapping detection boxes."""
    if not boxes:
        return []
    merged = []
    boxes = sorted(boxes, key=lambda b: b[0])
    current = boxes[0]
    for b in boxes[1:]:
        if abs(b[0]-current[0]) < 40 and abs(b[1]-current[1]) < 40:
            current = (
                min(current[0], b[0]),
                min(current[1], b[1]),
                max(current[0]+current[2], b[0]+b[2]) - min(current[0], b[0]),
                max(current[1]+current[3], b[1]+b[3]) - min(current[1], b[1])
            )
        else:
            merged.append(current)
            current = b
    merged.append(current)
    return merged

def smooth_box(history):
    """Returns averaged box across history."""
    xs = [b[0] for b in history]
    ys = [b[1] for b in history]
    ws = [b[2] for b in history]
    hs = [b[3] for b in history]
    return (int(np.mean(xs)), int(np.mean(ys)), int(np.mean(ws)), int(np.mean(hs)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # ✅ Stabilization
    # -----------------------------
    if prev_gray is not None:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=90, qualityLevel=0.2, minDistance=40)
        if prev_pts is not None:
            curr_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            idx = np.where(st == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]
            if len(prev_pts) > 12:
                m, _ = cv2.estimateAffine2D(prev_pts, curr_pts)
                if m is not None:
                    frame = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
    prev_gray = gray

    # -----------------------------
    # ✅ Noise reduction
    # -----------------------------
    filtered = cv2.GaussianBlur(frame, (5,5), 1)

    # -----------------------------
    # ✅ Multi-detector fusion
    # -----------------------------
    gray2 = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    detections = []
    detections.extend(cascade_body.detectMultiScale(gray2, scaleFactor=1.05, minNeighbors=4, minSize=(30, 60)))
    detections.extend(cascade_upper.detectMultiScale(gray2, scaleFactor=1.05, minNeighbors=4, minSize=(30, 60)))
    detections.extend(cascade_face.detectMultiScale(gray2, scaleFactor=1.05, minNeighbors=4, minSize=(30, 40)))

    # Merge overlapping detections
    detections = merge_boxes(detections)

    # -----------------------------
    # ✅ Temporal filtering
    # -----------------------------
    if detections:
        detection_history.append(detections[0])
        if len(detection_history) > HISTORY_SIZE:
            detection_history.pop(0)
        box = smooth_box(detection_history)
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
        cv2.putText(frame, "PERSON", (box[0], box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    else:
        # Clear history when detection drops completely
        detection_history.clear()

    cv2.imshow("Improved Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
