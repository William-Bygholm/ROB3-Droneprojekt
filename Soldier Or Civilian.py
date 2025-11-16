import cv2
import numpy as np

def classify_person(roi, threshold=0.3):
    roi = cv2.resize(roi, (64, 128))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()