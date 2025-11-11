
#cap = cv2.VideoCapture("C:/Users/alexa/Downloads/drive-download-20251110T110651Z-1-002/Militær uden bånd.MP4")
import cv2
import numpy as np

# ---------------------------------------------------
# HOG SETUP
# ---------------------------------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ---------------------------------------------------
# TRACKING STATE
# ---------------------------------------------------
tracked_people = []
track_id_counter = 0

# each entry = { "id": int, "x": int, "y": int, "w": int, "h": int,
#                "frames_since_seen": int, "history": int }

MAX_MISS = 15      # person disappears after 15 missed frames
MIN_CONFIRM = 2    # need 2 hits to activate a track
IOU_THRESHOLD = 0.35


# ---------------------------------------------------
# IOU calculation
# ---------------------------------------------------
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


# ---------------------------------------------------
# Track association
# ---------------------------------------------------
def match_tracks(detections):
    global tracked_people, track_id_counter

    # Mark all old tracks as unseen
    for t in tracked_people:
        t["frames_since_seen"] += 1

    # Try to match each detection with existing tracks
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
            # Update the matched track
            best_track["x"] = det[0]
            best_track["y"] = det[1]
            best_track["w"] = det[2]
            best_track["h"] = det[3]
            best_track["frames_since_seen"] = 0
            best_track["history"] += 1
        else:
            # Create new track
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

    # Remove dead tracks
    tracked_people = [t for t in tracked_people if t["frames_since_seen"] < MAX_MISS]


# ---------------------------------------------------
# HOG detection
# ---------------------------------------------------
def detect_people(gray):
    rects, weights = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    cleaned = []

    for (x, y, w, h) in rects:
        # Size filtering
        if w < 25 or h < 60:
            continue

        # Aspect ratio filtering
        ratio = h / float(w)
        if ratio < 1.2 or ratio > 4.0:
            continue

        # Keep box
        cleaned.append((x, y, w, h))

    return cleaned


# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
cap = cv2.VideoCapture("C:/Users/alexa/Downloads/drive-download-20251110T110651Z-1-002/Militær uden bånd.MP4")

TARGET_WIDTH = 640

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize video
    h, w = frame.shape[:2]
    scale = TARGET_WIDTH / w
    resized_h = int(h * scale)
    frame = cv2.resize(frame, (TARGET_WIDTH, resized_h))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detect_people(gray)
    match_tracks(detections)

    # Draw stable tracks
    for t in tracked_people:
        # Only display tracks that have been confirmed
        if t["history"] >= MIN_CONFIRM:
            cv2.rectangle(frame, 
                          (t["x"], t["y"]),
                          (t["x"] + t["w"], t["y"] + t["h"]),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"ID {t['id']}", 
                        (t["x"], t["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Stable People Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


