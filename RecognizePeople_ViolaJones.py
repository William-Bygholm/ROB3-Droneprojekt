import cv2
import imutils
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

cap = cv2.VideoCapture('ProjektVideoer/2 militær med blå bånd .MP4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=min(600, frame.shape[1]))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    faces = face_cascade.detectMultiScale(hsv, scaleFactor=1.1, minNeighbors=7)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Human", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Viola-Jones Human Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()