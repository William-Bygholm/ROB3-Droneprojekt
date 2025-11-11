import cv2
import numpy as np

video_path = "C:/Users/alexa/Downloads/Civil person.MP4"
resize_scale = 0.6
min_blob_area = 500
green_fraction_threshold = 0.25

cap = cv2.VideoCapture(video_path)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 244, 255, cv2.THRESH_BINARY)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # Convert ROI to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30,40,40])
        upper_green = np.array([85,255,255])
        green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)

        green_pixels = cv2.countNonZero(green_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        green_fraction = green_pixels / (total_pixels + 1e-6)

        # Decide MILITARY or CIVILIAN
        if green_fraction >= green_fraction_threshold:
            label = "MILITARY"
            color = (0,255,0)
        else:
            label = "CIVILIAN"
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({green_fraction:.2f})", (x, max(0,y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Result", frame)
    cv2.imshow("FG Mask", fgmask)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
