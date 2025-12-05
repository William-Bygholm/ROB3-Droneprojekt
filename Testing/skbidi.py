import cv2
import joblib
import json
import numpy as np

VIDEO = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 mili en idiot der ligger ned.MP4"
JSONFILE = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json"

# load frame 100
cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

clf_data = joblib.load(r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Person_Detector_Json+YOLO.pkl")
clf = clf_data.get("classifier", clf_data)

hog = cv2.HOGDescriptor((128,256),(32,32),(16,16),(8,8),9)

rects, weights = hog.detectMultiScale(gray, winStride=(24,24), padding=(0,0), scale=1.05)

print("HOG rects:", len(rects))
print("First 5 rects:", rects[:5])
print("First 5 scores:", weights[:5])

# load ground truth
coco = json.load(open(JSONFILE))
gt_boxes = [ann['bbox'] for ann in coco['annotations'] if ann['image_id']==100]
print("GT:", gt_boxes)
