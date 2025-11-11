import cv2
import numpy as np

# HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture("C:/Users/alexa/Downloads/drive-download-20251110T110651Z-1-002/Militær uden bånd.MP4")
resize_scale = 0.35

prev_gray = None
prev_pts = None

MAX_LOST = 25  # frames to keep memory
HISTORY_SIZE = 8

# Tracked people list
tracked_people = []  # each element: {'box':(x,y,w,h),'lost_counter':int,'history':[boxes]}

# ------------------------
# Helper functions
# ------------------------
def smooth_box(history):
    xs = [b[0] for b in history]
    ys = [b[1] for b in history]
    ws = [b[2] for b in history]
    hs = [b[3] for b in history]
    return (
        int(np.mean(xs)),
        int(np.mean(ys)),
        int(np.mean(ws)),
        int(np.mean(hs))
    )

def classify_shirt_color(roi):
    """Return 'MILITARY' if mostly green/gray, else 'CIVILIAN'"""
    if roi.size == 0:
        return "CIVILIAN"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # focus on hue, allow low saturation/brightness
    lower_hue = 25
    upper_hue = 85
    hue = hsv[:,:,0]
    mask = cv2.inRange(hue, lower_hue, upper_hue)
    green_ratio = cv2.countNonZero(mask) / (roi.shape[0]*roi.shape[1])
    return "MILITARY" if green_ratio > 0.3 else "CIVILIAN"


def iou(boxA, boxB):
    # Intersection over Union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

# ------------------------
# Main loop
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
    h, w = frame.shape[:2]
    stabilized = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------
    # Stabilization
    # --------------------------
    if prev_gray is not None:
        if prev_pts is None:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=40, qualityLevel=0.2, minDistance=30)
        if prev_pts is not None:
            curr_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            if curr_pts is not None:
                good_old = prev_pts[st == 1]
                good_new = curr_pts[st == 1]
                if len(good_old) > 10:
                    M, _ = cv2.estimateAffine2D(good_old, good_new)
                    if M is not None:
                        stabilized = cv2.warpAffine(frame, M, (w, h))
                prev_pts = good_new.reshape(-1,1,2)
    prev_gray = gray

    # --------------------------
    # Noise reduction
    # --------------------------
    filtered = cv2.GaussianBlur(stabilized, (3,3), 0)

    # --------------------------
    # HOG detection
    # --------------------------
    boxes, weights = hog.detectMultiScale(filtered, winStride=(8,8), padding=(8,8), scale=1.05)
    filtered_boxes = []
    for box, weight in zip(boxes, weights):
        if weight>0.6:
            x,y,w_box,h_box = box
            if 30<w_box<350 and 40<h_box<350:
                filtered_boxes.append((x,y,w_box,h_box))

    # --------------------------
    # Update tracked people
    # --------------------------
    updated_tracked = []
    used_detections = set()
    for person in tracked_people:
        best_match = None
        best_iou = 0
        for i, det in enumerate(filtered_boxes):
            if i in used_detections:
                continue
            score = iou(person['box'], det)
            if score > best_iou:
                best_iou = score
                best_match = i
        if best_iou > 0.3:
            # Match found, update box
            det = filtered_boxes[best_match]
            used_detections.add(best_match)
            person['box'] = det
            person['lost_counter'] = 0
            person['history'].append(det)
            if len(person['history'])>HISTORY_SIZE:
                person['history'].pop(0)
        else:
            # No match, increase lost counter
            person['lost_counter'] +=1
        if person['lost_counter']<MAX_LOST:
            updated_tracked.append(person)
    tracked_people = updated_tracked

    # Add unmatched detections as new people
    for i, det in enumerate(filtered_boxes):
        if i not in used_detections:
            tracked_people.append({'box':det,'lost_counter':0,'history':[det]})

    # --------------------------
    # Draw all tracked people
    # --------------------------
    output = frame.copy()
    for person in tracked_people:
        smoothed = smooth_box(person['history'])
        x,y,w_box,h_box = smoothed
        shirt_roi = frame[y:y+int(h_box/2), x:x+w_box]
        shirt_label = classify_shirt_color(shirt_roi)
        cv2.rectangle(output, (x,y), (x+w_box,y+h_box), (0,255,0),2)
        cv2.putText(output, f"{shirt_label}", (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("HOG Multi-Person + Shirt Color", output)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
