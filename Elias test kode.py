import cv2
import numpy as np
from sklearn.svm import LinearSVC
from skimage.feature import hog
import joblib
import os

# -----------------------------
# Load / Train SVM model
# -----------------------------
SVM_MODEL_PATH = "hog_svm_model.pkl"

if os.path.exists(SVM_MODEL_PATH):
    print("Loading pretrained SVM model...")
    svm = joblib.load(SVM_MODEL_PATH)
else:
    print("No pretrained SVM found. You need to train on positive/negative HOG samples.")
    raise RuntimeError("Please train your SVM first.")

# -----------------------------
# Initialize HOG descriptor
# -----------------------------
hog_descriptor = cv2.HOGDescriptor()

# -----------------------------
# Video
# -----------------------------
cap = cv2.VideoCapture("ProjektVideoer/Civil person.MP4")
resize_scale = 0.35

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # Candidate windows
    # -----------------------------
    boxes, weights = hog_descriptor.detectMultiScale(frame_resized,
                                                     winStride=(8,8),
                                                     padding=(8,8),
                                                     scale=1.05)
    svm_boxes = []
    for (x, y, w_box, h_box) in boxes:
        window = gray[y:y+h_box, x:x+w_box]
        # resize to fixed size
        window_resized = cv2.resize(window, (64, 128))
        # HOG feature
        feature = hog(window_resized,
                      orientations=9,
                      pixels_per_cell=(8,8),
                      cells_per_block=(2,2),
                      block_norm='L2-Hys',
                      feature_vector=True)
        # SVM prediction
        pred = svm.predict([feature])
        if pred[0] == 1:
            svm_boxes.append((x, y, w_box, h_box))

    # -----------------------------
    # Draw boxes
    # -----------------------------
    output = frame_resized.copy()
    for (x, y, w_box, h_box) in svm_boxes:
        cv2.rectangle(output, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
        cv2.putText(output, "PERSON (SVM)", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("HOG + SVM Detection", output)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
