import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

"""
# Read image and resize it
img = cv2.imread('Billeder/IMG_20251104_094347.jpg')
resized = cv2.resize(img, None, fx=0.2, fy=0.2)

# Find people in the image
boxes, weights = hog.detectMultiScale(resized, winStride=(4,4), padding=(8,8), scale=1.02)

# Draw bounding boxes around detected people
for (x, y, w, h) in boxes:
    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Show result
cv2.imshow('HOG People Detection', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
cap = cv2.VideoCapture('ProjektVideoer/Militær blåt bånd.MP4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    resized = cv2.resize(frame, None, fx=0.2, fy=0.2)
    
    boxes, weights = hog.detectMultiScale(resized, winStride=(4,4), padding=(4,4), scale=1.05)

    for (x, y, w, h) in boxes:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('HOG People Detection', resized)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()