import cv2
import numpy as np
import glob
import os

# ===============================
# SETTINGS
# ===============================
VIDEO_PATH = "C:/Users/alexa/Downloads/drive-download-20251110T110651Z-1-002/Militær uden bånd.MP4"
SCALE = 0.4

SILHOUETTE_TEMPLATES_DIR = "C:/Users/alexa/Desktop/Video til Projekt/Olaf Canny edge.png"  # Edge silhouettes
MATCH_THRESHOLD = 0.2
ASPECT_RANGE = (1.5, 4.5)
MIN_AREA = 500
MAX_AREA = 120000
MERGE_DIST = 20  # distance in pixels to merge nearby contours
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

# Load silhouette templates
templates = []
for p in glob.glob(os.path.join(SILHOUETTE_TEMPLATES_DIR, "*.jpg")):
    t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if t is None:
        continue
    t = cv2.Canny(t, 50, 150)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        templates.append(max(cnts, key=cv2.contourArea))

# ===============================
# HELPER FUNCTIONS
# ===============================
def merge_contours(contours, dist_thresh=MERGE_DIST):
    """Merge contours that are within dist_thresh pixels of each other"""
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
            # if boxes are close
            if (abs(x1 - x2) < dist_thresh and abs(y1 - y2) < dist_thresh) or \
               (abs((x1+w1) - (x2+w2)) < dist_thresh and abs((y1+h1)-(y2+h2)) < dist_thresh):
                # merge
                rx1 = min(rx1, x2)
                ry1 = min(ry1, y2)
                rw1 = max(rx1 + rw1, x2 + w2) - rx1
                rh1 = max(ry1 + rh1, y2 + h2) - ry1
                used[j] = True
        merged.append((rx1, ry1, rw1, rh1))
        used[i] = True
    return merged

def region_template_match(frame, bbox, templates):
    """Compute the best matchShapes score between the edge patch and templates"""
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return 1.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    cand = max(cnts, key=cv2.contourArea)
    best_score = 1.0
    for t in templates:
        try:
            score = cv2.matchShapes(cand, t, cv2.CONTOURS_MATCH_I1, 0.0)
        except:
            score = 1.0
        best_score = min(best_score, score)
    return best_score

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # 3) Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if MIN_AREA <= cv2.contourArea(c) <= MAX_AREA]

    # 4) Merge nearby contours to get candidate regions
    candidates = merge_contours(contours)

    # 5) Filter candidates based on aspect ratio and template matching
    detections = []
    for (x, y, w, h) in candidates:
        aspect = float(h)/float(w)
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            continue
        score = region_template_match(frame, (x,y,w,h), templates)
        if score <= MATCH_THRESHOLD:
            detections.append((x, y, w, h))

    # 6) Temporal persistence filter
    new_tracks = []
    for (x, y, w, h) in detections:
        matched = False
        for tx, ty, tw, th, count in tracks:
            if abs(x - tx) < 20 and abs(y - ty) < 20:
                new_tracks.append((x, y, w, h, count+1))
                matched = True
                break
        if not matched:
            new_tracks.append((x, y, w, h, 1))
    tracks = new_tracks
    confirmed = [(x,y,w,h) for (x,y,w,h,c) in tracks if c >= PERSISTENCE_FRAMES]

    # 7) Draw detections
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
