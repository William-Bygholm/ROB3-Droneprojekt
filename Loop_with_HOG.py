import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("C:/Users/alexa/Downloads/Civil person.MP4")

# Load detector
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Parameters
resize_scale = 0.5
prev_gray = None
transforms = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize
    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # ✅ 1. STABILIZATION
    # -----------------------------
    if prev_gray is not None:
        # Detect feature points
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=30)
        
        if prev_pts is not None:
            # Track feature points
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

            # Keep valid pairs
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            if len(prev_pts) > 10:
                # Estimate affine transformation
                m, _ = cv2.estimateAffine2D(prev_pts, curr_pts)
                if m is not None:
                    # Warp frame to reduce drone movement
                    stabilized = cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]))
                else:
                    stabilized = frame.copy()
            else:
                stabilized = frame.copy()
        else:
            stabilized = frame.copy()
    else:
        stabilized = frame.copy()

    prev_gray = gray

    # -----------------------------
    # ✅ 2. NOISE REDUCTION
    # -----------------------------
    denoised = cv2.bilateralFilter(stabilized, 7, 50, 50)

    # -----------------------------
    # ✅ 3. CONTRAST NORMALIZATION
    # -----------------------------
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -----------------------------
    # ✅ 4. EDGE ENHANCEMENT
    # -----------------------------
    edges = cv2.Canny(cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY), 100, 200)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8), iterations=1)

    # -----------------------------
    # ✅ 5. HUMAN DETECTION
    # -----------------------------
    gray_filtered = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

    humans = cascade.detectMultiScale(
        gray_filtered,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 60)
    )

    for (x,y,w,h) in humans:
        cv2.rectangle(stabilized, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(stabilized, "Person", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # -----------------------------
    # ✅ SHOW RESULTS
    # -----------------------------
    cv2.imshow("Stabilized", stabilized)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
