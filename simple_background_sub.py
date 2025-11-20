import cv2
import numpy as np
from pathlib import Path

# # Define the folder path
# folder_path = Path("Billeder")
#
# # Check if the folder exists
# if not folder_path.exists():
#     raise FileNotFoundError(f"The folder {folder_path} does not exist.")
#
# # List all image files in the folder
# image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
# if not image_files:
#     raise FileNotFoundError(f"No image files found in the folder {folder_path}.")
# print(f"Found {len(image_files)} image files.")

# Initialize video capture and background subtractor
cap = cv2.VideoCapture("ProjektVideoer/2 mili der ligger ned og 1 civil.MP4")  # Use your drone's video feed
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break # no more frames or failed to read frame

    # blur
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # apply background subtraction
    fgmask = fgbg.apply(blur, learningRate=0.01)

    # MOG2 marks shadows with value 127 — keep only definite foreground (255)
    fgmask_bin = np.where(fgmask == 255, 255, 0).astype('uint8')
    
    ## optional: clean up mask (stronger morphology)
    _, thresh = cv2.threshold(fgmask_bin, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    # find contours (compatible with different OpenCV return signatures)
    contours_info = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    # draw bounding boxes for sufficiently large contours (filter small/huge noise)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 40000 < area < (frame.shape[0] * frame.shape[1] * 0.5):  # tune limits
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # --- scale for display only ---
    max_width, max_height = 960, 720  # change to desired display max size
    h, w = frame.shape[:2]
    scale = min(1.0, max_width / w, max_height / h)
    if scale < 1.0:
        disp_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        disp_mask = cv2.resize(fgmask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        disp_frame = frame
        disp_mask = fgmask

    cv2.imshow('Frame', disp_frame)
    cv2.imshow('FG Mask', disp_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
