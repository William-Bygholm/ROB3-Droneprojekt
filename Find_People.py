import cv2
import numpy as np
import glob
import os

# ===============================
# SETTINGS
# ===============================
VIDEO_PATH = "C:/Users/alexa/Downloads/drive-download-20251110T110651Z-1-002/Militær uden bånd.MP4"
SCALE = 0.5
SILHOUETTE_TEMPLATES_DIR = "C:/Users/alexa/Desktop/Video til Projekt/Olaf template .png"  # folder with silhouette edge images
MATCH_THRESHOLD = 0.15
ASPECT_RANGE = (1.8, 4.5)
MIN_AREA = 600
MAX_AREA = 90000
SYMMETRY_THRESHOLD = 0.4
PERSISTENCE_FRAMES = 3
DISPLAY = True

# ===============================
# INIT
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
    if t is None: continue
    t = cv2.Canny(t, 50, 150)
    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        templates.append(max(cnts, key=cv2.contourArea))

# ===============================
# HELPER FUNCTIONS
# ===============================
def symmetry_score(contour):
    """Compute horizontal symmetry score of contour"""
    x, y, w, h = cv2.boundingRect(contour)
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(roi, [contour - [x, y]], -1, 255, -1)

    mid = w // 2
    left = roi[:, :mid]
    right = np.fliplr(roi[:, mid:])

    # Ensure same shape
    min_w = min(left.shape[1], right.shape[1])
    left = left[:, :min_w]
    right = right[:, :min_w]

    inter = np.sum(np.logical_and(left > 0, right > 0))
    union = np.sum(np.logical_or(left > 0, right > 0))

    if union == 0:
        return 0.0
    return inter / float(union)

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Stabilization (to reduce camera motion)
    if prev_gray is not None:
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(blur, None)
        if des1 is not None and des2 is not None and len(kp1) > 8 and len(kp2) > 8:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:300]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)
            if M is not None:
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    prev_gray = blur.copy()

    # 2. Edge detection
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # 3. Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / float(w)
        if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
            continue
        sym = symmetry_score(c)
        if sym < SYMMETRY_THRESHOLD:
            continue

        # Compare to silhouette templates
        best_match = 1.0
        for t in templates:
            score = cv2.matchShapes(c, t, cv2.CONTOURS_MATCH_I1, 0.0)
            best_match = min(best_match, score)
        if best_match > MATCH_THRESHOLD:
            continue

        detections.append((x, y, w, h))

    # 4. Temporal persistence filter
    new_tracks = []
    for (x, y, w, h) in detections:
        matched = False
        for tx, ty, tw, th, count in tracks:
            if abs(x - tx) < 20 and abs(y - ty) < 20:
                new_tracks.append((x, y, w, h, count + 1))
                matched = True
                break
        if not matched:
            new_tracks.append((x, y, w, h, 1))
    tracks = new_tracks
    confirmed = [(x, y, w, h) for (x, y, w, h, c) in tracks if c >= PERSISTENCE_FRAMES]

    # 5. Display
    output = frame.copy()
    for (x, y, w, h) in confirmed:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if DISPLAY:
        combo = np.hstack([cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), output])
        cv2.imshow("Edges | Detections", combo)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
