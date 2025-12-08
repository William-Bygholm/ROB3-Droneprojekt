import cv2
import joblib
import numpy as np
import json
import os
import time
from collections import defaultdict
from Main import (
    WINDOW_SIZE, SCALES, STEP_SIZES, NMS_THRESHOLD, SVM_THRESHOLD, FRAME_SKIP,
    clf, hog,
    detect_people, crop_image, blob_analysis, classify_target,
    load_reference_histograms, classify_person
)

# ---------------- USER CONFIG ----------------
VIDEO_FILES = [
    r"ProjektVideoer/2 mili en idiot der ligger ned.MP4",
    r"ProjektVideoer/3 mili 2 onde 1 god.MP4"
]

JSON_FILES = [
    r"Testing/2 mili og 1 idiot.json",
    r"Testing/3mili 2 onde 1 god.json"
]

REFERENCE_DIR = "Reference templates"
reference_histograms = load_reference_histograms(REFERENCE_DIR)

# Alle klasser inkl. HVT, area-based og Unidentified
CLASSES = [
    "Good soldier", "Bad soldier",
    "Good soldier (HVT)", "Bad soldier (HVT)",
    "Good soldier (area-based)", "Bad soldier (area-based)",
    "Good soldier (area-based, 2v2)", "Bad soldier (area-based, 2v2)",
    "Unidentified soldier", "Civilian"
]

# Confusion matrix initialization
conf_matrix = {pred: {gt: 0 for gt in CLASSES} for pred in CLASSES}
false_negatives_count = defaultdict(int)

# ---------------- HELPER FUNCTIONS ----------------
def load_json_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def match_detection_to_groundtruth(det_box, gt_boxes):
    x1, y1, x2, y2 = det_box
    best_iou = 0
    best_label = None
    for gt in gt_boxes:
        gx1, gy1, gx2, gy2 = gt["bbox"]
        inter_x1 = max(x1, gx1)
        inter_y1 = max(y1, gy1)
        inter_x2 = min(x2, gx2)
        inter_y2 = min(y2, gy2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_det = (x2 - x1) * (y2 - y1)
        area_gt = (gx2 - gx1) * (gy2 - gy1)
        iou = inter_area / float(area_det + area_gt - inter_area) if (area_det + area_gt - inter_area) > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_label = gt["label"]
    if best_iou < 0.5:
        return None
    return best_label

def classify_detection(roi):
    classification, _ = classify_person(roi, reference_histograms, threshold_score=0.8)
    if classification == "Civilian":
        return "Civilian"
    cropped_roi = crop_image(roi)
    blurred = cv2.GaussianBlur(cropped_roi, (5,5), 0)
    _, _, _, red_boxes, blue_boxes = blob_analysis(blurred, morph_kernel=(3,3), morph_iters=1)
    target_class, target_type, is_hvt = classify_target(red_boxes, blue_boxes)
    if target_class is None:
        return "Unidentified soldier"
    return target_class

# ---------------- PROCESSING ----------------
total_frames = sum(int(cv2.VideoCapture(v).get(cv2.CAP_PROP_FRAME_COUNT)) for v in VIDEO_FILES)

processed_frames = 0
start_time = time.time()

for video_file, json_file in zip(VIDEO_FILES, JSON_FILES):
    gt_boxes = load_json_labels(json_file)
    
    cap = cv2.VideoCapture(video_file)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue
        processed_frames += 1
        
        elapsed = time.time() - start_time
        eta = (elapsed / processed_frames) * (total_frames/FRAME_SKIP - processed_frames)
        
        # Detect people
        detections = detect_people(frame, clf, hog)
        
        # Classify detections
        for det_box in detections:
            x1, y1, x2, y2 = det_box
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            pred_class = classify_detection(roi)
            gt_label = match_detection_to_groundtruth(det_box, gt_boxes)
            if gt_label is None:
                gt_label = "Unidentified soldier"
            
            conf_matrix[pred_class][gt_label] += 1
        
        # False negatives
        gt_labels_in_frame = [gt["label"] for gt in gt_boxes]
        for gt_label in gt_labels_in_frame:
            detected = any(match_detection_to_groundtruth(det_box, gt_boxes) == gt_label for det_box in detections)
            if not detected:
                false_negatives_count[gt_label] += 1
        
        # Løbende print
        print(f"\nProcessed frames: {processed_frames}, ETA: {eta:.1f}s")
        print("Detection counts per class:")
        for cls in CLASSES:
            det_count = sum(conf_matrix[cls][gt] for gt in CLASSES)
            print(f" {cls}: {det_count}")
        print("\nFalse negatives per class:")
        for cls in CLASSES:
            print(f" {cls}: {false_negatives_count[cls]}")
        print("-"*50)
    
    cap.release()

# ---------------- FINAL RESULTS ----------------
print("\nFinal Confusion Matrix (Predicted x GroundTruth):")
header = "\t" + "\t".join(CLASSES)
print(header)
for pred in CLASSES:
    row = pred + "\t" + "\t".join(str(conf_matrix[pred][gt]) for gt in CLASSES)
    print(row)

print("\nFinal False Negatives per class:")
for cls in CLASSES:
    print(f"{cls}: {false_negatives_count[cls]}")
