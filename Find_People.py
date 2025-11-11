import cv2
import numpy as np

# -------------------------------
# HOG setup
# -------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# -------------------------------
# Tracking state
# -------------------------------
tracked_people = []
track_id_counter = 0

# Each entry:
# {"id": int, "x": int, "y": int, "w": int, "h": int, "frames_since_seen": int, "history": int}
MAX_MISS = 15
MIN_CONFIRM = 2
IOU_THRESHOLD = 0.35

# -------------------------------
# IOU calculation
# -------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

# -------------------------------
# Track association
# -------------------------------
def match_tracks(detections):
    global tracked_people, track_id_counter

    for t in tracked_people:
        t["frames_since_seen"] += 1

    for det in detections:
        best_iou = 0
        best_track = None

        for t in tracked_people:
            test_box = (t["x"], t["y"], t["w"], t["h"])
            score = iou(test_box, det)
            if score > best_iou:
                best_iou = score
                best_track = t

        if best_iou > IOU_THRESHOLD:
            best_track["x"] = det[0]
            best_track["y"] = det[1]
            best_track["w"] = det[2]
            best_track["h"] = det[3]
            best_track["frames_since_seen"] = 0
            best_track["history"] += 1
        else:
            tracked_people.append({
                "id": track_id_counter,
                "x": det[0],
                "y": det[1],
                "w": det[2],
                "h": det[3],
                "frames_since_seen": 0,
                "history": 1
            })
            track_id_counter += 1

    tracked_people[:] = [t for t in tracked_people if t["frames_since_seen"] < MAX_MISS]

# -------------------------------
# HOG detection
# -------------------------------
def detect_people(gray):
    rects, weights = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    cleaned = []
    for (x, y, w, h) in rects:
        if w < 25 or h < 60:
            continue
        ratio = h / float(w)
        if ratio < 1.2 or ratio > 4.0:
            continue
        cleaned.append((x, y, w, h))
    return cleaned

# -------------------------------
# Main loop
# -------------------------------
cap = cv2.VideoCapture("C:/Users/alexa/Downloads/drive-download-20251110T110651Z-1-002/Militær uden bånd.MP4")
DISPLAY_WIDTH = 640

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Keep original frame for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people at original resolution
    detections = detect_people(gray)

    # Match detections to existing tracks
    match_tracks(detections)

    # Draw confirmed tracks
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])))
    scale_x = DISPLAY_WIDTH / frame.shape[1]
    scale_y = display_frame.shape[0] / frame.shape[0]

    for t in tracked_people:
        if t["history"] >= MIN_CONFIRM:
            x1 = int(t["x"] * scale_x)
            y1 = int(t["y"] * scale_y)
            x2 = int((t["x"] + t["w"]) * scale_x)
            y2 = int((t["y"] + t["h"]) * scale_y)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"ID {t['id']}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("People Detection Stable", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
