import cv2
import numpy as np
import os

def compute_histogram(img, target_size=(64, 128), height_ratio=0.3, vertical_center=0.4, width_ratio=0.7):
    """
    Help-function to compute a normalized HSV histogram for the upper part of an image.
    This is used both for reference histograms creation and for classification.
    """
    img = cv2.resize(img, target_size)
    h, w = img.shape[:2]

    new_h = int(h*height_ratio)
    y_center = int(h*vertical_center)
    y_start = max(0, y_center - new_h // 2)
    y_end = min(h, y_start + new_h)

    new_w = int(w*width_ratio)
    x_start = max(0, (w - new_w) // 2)
    x_end = min(w, x_start + new_w)

    cropped = img[y_start:y_end, x_start:x_end]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def load_reference_histograms(base_dir):
    reference_histograms = {}
    for label in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, label)
        if not os.path.isdir(class_dir):
            continue
        histograms = []
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg')):
                path = os.path.join(class_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    histograms.append(compute_histogram(img))
        reference_histograms[label] = histograms
    return reference_histograms