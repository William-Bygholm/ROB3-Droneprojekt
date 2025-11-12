import cv2
import numpy as np

# ===============================
# SETTINGS
# ===============================
VIDEO_PATH = "C:/Users/alexa/Downloads/drive-download-20251110T110651Z-1-002/Militær uden bånd.MP4"
SCALE = 0.4

MIN_AREA = 500
MAX_AREA = 120000
ASPECT_RANGE = (1.5, 4.5)
EDGE_DENSITY_THRESHOLD = 0.05  # fraction of pixels in candidate box that are edges
MERGE_DIST = 20
PERSISTENCE_FRAMES = 3
DISPLAY = True

# ===============================
# INITIALIZATION
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
orb = cv2.ORB_create(1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
prev_gray = None
tracks = []

# ===============================
# HELPER FUNCTIONS
# ===============================
def merge_contours(contours, dist_thresh=MERGE_DIST):
    """Merge contours that are within dist_thresh pixels"""
    bboxes = [cv2.boundingRect(c) for c in contours]
    merged = []
    used = [False]*len(bboxes)

    for i, (x1, y1, w1, h1) in enumerate(bboxes):
        if used[i]:
            continue
        rx1, ry1, rw1, rh1 = x1, y1, w1, h1
        for j, (x2, y2, w2, h2) in enumerate(bboxes):
            if i == j or used[j]:
                continue
            # merge if close
            if abs(x1 - x2) < dist_thresh and abs(y1 - y2) < dist_thresh:
                rx1 = min(rx1, x2)
                ry1 = min(ry1, y2)
                rw1 = max(rx1 + rw1, x2 + w2) - rx1
                rh1 = max(ry1 + rh1, y2 + h2) - ry1
                used[j] = True
        merged.append((rx1, ry1, rw1, rh1))
        used[i] = True
    return merged

def candidate_score(edges, bbox):
    """Compute edge density inside bbox"""
    x, y, w, h = bbox
    roi = edges[y:y+h, x:x+w]
    if roi.size == 0:
        return 0
    edge_fraction = np.count_nonzero(roi) / float(roi.size)
    return edge_fraction

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 1) Stabilization
    if prev_gray is not None:
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(blur, None)
        if des1 is not None and des2 is not None and len(kp1)>8 and len(kp2)>8:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:300]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
            if M is not None:
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    prev_gray = blur.copy()

    # 2) Edge detection
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    # 3) Find contours (all fragments)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if MIN_AREA <= cv2.contourArea(c) <= MAX_AREA]

    # 4) Merge nearby contours
    candidates = merge_contours(contours)

    # 5) Score candidates (aspect ratio + edge density)
    detections = []
    for (x,y,w,h) in candidates:
        aspect = float(h)/float(w)
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            continue
        score = candidate_score(edges, (x,y,w,h))
        if score >= EDGE_DENSITY_THRESHOLD:
            detections.append((x,y,w,h))

    # 6) Temporal persistence
    new_tracks = []
    for (x,y,w,h) in detections:
        matched = False
        for tx, ty, tw, th, count in tracks:
            if abs(x - tx) < 20 and abs(y - ty) < 20:
                new_tracks.append((x,y,w,h,count+1))
                matched = True
                break
        if not matched:
            new_tracks.append((x,y,w,h,1))
    tracks = new_tracks
    confirmed = [(x,y,w,h) for (x,y,w,h,c) in tracks if c >= PERSISTENCE_FRAMES]

    # 7) Draw
    output = frame.copy()
    for (x,y,w,h) in confirmed:
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(output, "Person", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

    if DISPLAY:
        combo = np.hstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), output])
        cv2.imshow("Edges | Detections", combo)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
