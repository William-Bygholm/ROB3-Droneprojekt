import cv2
import numpy as np

# ----------------------------------------
# 1. Load Haar Cascade classifier
# ----------------------------------------
# OpenCV ships with several cascade files; fullbody works for standing people
# Other options: haarcascade_upperbody.xml, haarcascade_lowerbody.xml
cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
cascade = cv2.CascadeClassifier(cascade_path)

if cascade.empty():
    raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")

print(f"Loaded Haar cascade: {cascade_path}")

# ----------------------------------------
# 2. Load video
# ----------------------------------------
cap = cv2.VideoCapture("ProjektVideoer\Civil person.MP4")
resize_scale = 0.5  # adjust for speed vs accuracy

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------
    # 3. Detect people using Haar cascade
    # ----------------------------------------
    # Parameters to tune:
    # - scaleFactor: how much image is reduced at each scale (1.05–1.3)
    # - minNeighbors: how many neighbors each rect should have (3–6)
    # - minSize: minimum object size (width, height)
    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # smaller = more thorough but slower
        minNeighbors=3,       # higher = fewer false positives
        minSize=(30, 90),     # adjust based on expected person size in frame
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # ----------------------------------------
    # 4. Draw bounding boxes
    # ----------------------------------------
    output = frame.copy()
    for (x, y, w, h) in detections:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, "PERSON", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show detection count
    cv2.putText(output, f"Frame {frame_count} | Detections: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Haar Cascade Person Detection", output)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames, detections: {len(detections)}")

cap.release()
cv2.destroyAllWindows()