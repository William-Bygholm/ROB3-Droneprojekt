import cv2
import numpy as np
import os

def compute_histogram(img, center_y_ratio=0.35, center_x_ratio=0.5, height_ratio=0.2, width_ratio=0.3):
    """
    Help-function to compute a normalized HSV histogram for the upper part (breast region) of an image.
    This is used both for reference histograms creation and for classification.
    """
    h, w = img.shape[:2]

    new_h = max(1, int(h*height_ratio))
    y_center = int(h*center_y_ratio)
    y_start = max(0, y_center - (new_h // 2))
    y_end = min(h, y_start + new_h)

    new_w = max(1, int(w*width_ratio))
    x_center = int(w*center_x_ratio)
    x_start = max(0, x_center - (new_w // 2))
    x_end = min(w, x_start + new_w)

    cropped = img[y_start:y_end, x_start:x_end]
    if cropped.size == 0:
        raise ValueError("Cropped region has zero size. Check the cropping parameters.")
    
    # hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    # hist = cv2.normalize(hist, hist).astype("float32")
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2Lab)
    hist = cv2.calcHist([lab], [1, 2], None, [50, 60], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).astype("float32")
    return hist

def load_reference_histograms(base_dir):
    """
    A function to load reference images from a folder, compute their histograms, and store them in a dictionary.
    """
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

def show_crop_overlay(img, center_y_ratio=0.35, center_x_ratio=0.5, height_ratio=0.2, width_ratio=0.3):
    """
    A function only to test and visualize the cropping area used in compute_histogram.
    """
    h, w = img.shape[:2]

    crop_h = max(1, int(h * height_ratio))
    crop_w = max(1, int(w * width_ratio))
    y_center = int(h * center_y_ratio)
    x_center = int(w * center_x_ratio)

    y_start = max(0, y_center - crop_h // 2)
    y_end   = min(h, y_start + crop_h)
    x_start = max(0, x_center - crop_w // 2)
    x_end   = min(w, x_start + crop_w)

    # Tegn grøn boks
    overlay = img.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.imshow("Crop Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def classify_person(roi, reference_histograms, method=cv2.HISTCMP_BHATTACHARYYA, threshold_score=0.5):
    """
    Classify a person in the ROI as 'soldier' or 'unkown' based on histogram comparison.
    """
    roi_hist = compute_histogram(roi)
    best_score = float('inf')

    for histograms in reference_histograms.values():
        for ref_hist in histograms:
            score = cv2.compareHist(roi_hist, ref_hist, method)
            if score < best_score:
                best_score = score
    
    # Reverse logic for Bhattacharyya distance: lower=better to higher=better and normalize to [0, 1]:
    match_score = max(0.0, min(1.0, 1.0 - best_score))

    if best_score < threshold_score:
        print(f"Best score {best_score}")
        print(f"Classification: Military")
        return "Military", match_score
    else:
        print(f"No military match found. Best score: {best_score}")
        print(f"Classification: Civilian")
        return "Civilian", match_score

roi = cv2.imread('Billeder/Military long range 2.png')
reference_histograms = load_reference_histograms("Reference templates")
classification = classify_person(roi, reference_histograms)
show_crop_overlay(roi)