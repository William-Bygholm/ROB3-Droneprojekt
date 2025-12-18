"""
Full Detection + Classification Pipeline Test
Evaluates against COCO JSON ground truth annotations
Outputs: Confusion Matrix + False Negatives (combined for all videos)
"""

import cv2
import joblib
import numpy as np
import json
import os
import time
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from Main import (
    clf, hog, detect_people, crop_image, blob_analysis, classify_target,
    load_reference_histograms, classify_person, FRAME_SKIP
)

# ============ CONFIG ============
TEST_VIDEOS = [
    {
        "video": r"ProjektVideoer/2 mili en idiot der ligger ned.MP4",
        "json": r"Testing/2 mili og 1 idiot.json"
    },
    {
        "video": r"ProjektVideoer/3 mili 2 onde 1 god.MP4",
        "json": r"Testing/3mili 2 onde 1 god.json"
    }
]

REFERENCE_DIR = "Reference templates"
IOU_THRESHOLD = 0.3  # For matching detections to GT

# Class mapping - only the actual classes that exist
CLASS_LABELS = [
    "Good soldier",
    "Bad soldier", 
    "Good HVT",
    "Bad HVT",
    "Unidentified soldier",
    "Civilian"
]

# False Positive is tracked separately (detection with no matching GT)
FP_LABEL = "False Positive"

# ============ HELPERS ============

def extract_frame_number(filename):
    """Extract frame number from 'frame_000123.png' -> 123"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def load_coco_annotations(json_path):
    """
    Load COCO JSON and return frame-indexed ground truth boxes with labels.
    Returns: dict { frame_number: [ {"bbox": [x1,y1,x2,y2], "label": str}, ... ] }
    """
    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    # Map image_id to frame number
    image_id_to_frame = {}
    for img in coco.get("images", []):
        img_id = img["id"]
        frame_num = extract_frame_number(img["file_name"])
        image_id_to_frame[img_id] = frame_num
    
    # Build frame -> boxes mapping
    frame_to_boxes = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        frame_num = image_id_to_frame.get(img_id)
        if frame_num is None:
            continue
        
        # Extract bbox [x, y, w, h] -> [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        # Extract label from attributes
        attrs = ann.get("attributes", {})
        label = extract_label_from_attributes(attrs)
        
        frame_to_boxes[frame_num].append({
            "bbox": [x1, y1, x2, y2],
            "label": label
        })
    
    return frame_to_boxes

def extract_label_from_attributes(attrs):
    """Convert COCO attributes dict to our class label"""
    if attrs.get("Civilian"):
        return "Civilian"
    if attrs.get("Good HVT"):
        return "Good HVT"
    if attrs.get("Bad HVT"):
        return "Bad HVT"
    if attrs.get("Military good"):
        return "Good soldier"
    if attrs.get("Military bad"):
        return "Bad soldier"
    # Fallback
    return "Unidentified soldier"

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def classify_detection(roi, reference_histograms):
    """
    Full classification pipeline: civilian check + armband analysis
    Returns: class label (str)
    """
    # Step 1: Check if civilian
    classification, _ = classify_person(roi, reference_histograms, threshold_score=0.8)
    if classification == "Civilian":
        return "Civilian"
    
    # Step 2: Military -> analyze armbands
    cropped_roi = crop_image(roi)
    blurred = cv2.GaussianBlur(cropped_roi, (5, 5), 0)
    _, _, _, red_boxes, blue_boxes = blob_analysis(blurred, morph_kernel=(3, 3), morph_iters=1)
    
    target_class, target_type, is_hvt = classify_target(red_boxes, blue_boxes)
    
    if target_class in [None, "No target", "Uncertain classification"]:
        return "Unidentified soldier"
    
    # Map the output from classify_target to our 6 class labels
    # classify_target returns:
    # - "Good soldier", "Bad soldier" (1 armband)
    # - "Good soldier (HVT)", "Bad soldier (HVT)" (2 armbands same color)
    # - "Good soldier (area-based)", "Bad soldier (area-based)" (1v1 mixed)
    # - "Good soldier (area-based, 2v2)", "Bad soldier (area-based, 2v2)" (2v2 mixed)
    
    # Check if HVT (either from flag or from string)
    if is_hvt or "HVT" in target_class:
        if "Good" in target_class:
            return "Good HVT"
        elif "Bad" in target_class:
            return "Bad HVT"
    
    # Map all non-HVT variants to basic soldier classes
    if "Good" in target_class:
        return "Good soldier"
    elif "Bad" in target_class:
        return "Bad soldier"
    
    # Fallback
    return "Unidentified soldier"

def match_detections_to_gt(detections, gt_boxes, iou_threshold=0.5):
    """
    Match detections to ground truth using greedy IoU matching.
    Returns: 
        matched_pairs: [(det_idx, gt_idx), ...]
        unmatched_dets: [det_idx, ...]
        unmatched_gts: [gt_idx, ...]
    """
    matched_pairs = []
    matched_dets = set()
    matched_gts = set()
    
    # Build IoU matrix
    iou_matrix = []
    for i, det_box in enumerate(detections):
        row = []
        for j, gt in enumerate(gt_boxes):
            iou_val = compute_iou(det_box, gt["bbox"])
            row.append((iou_val, i, j))
        iou_matrix.extend(row)
    
    # Sort by IoU descending
    iou_matrix.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching
    for iou_val, i, j in iou_matrix:
        if iou_val < iou_threshold:
            break
        if i in matched_dets or j in matched_gts:
            continue
        matched_pairs.append((i, j))
        matched_dets.add(i)
        matched_gts.add(j)
    
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]
    
    return matched_pairs, unmatched_dets, unmatched_gts

# ============ MAIN PROCESSING ============

def main():
    print("=" * 80)
    print("FULL PIPELINE TEST - Detection + Classification vs Ground Truth")
    print("=" * 80)
    
    # Load reference histograms once
    print("\nLoading reference histograms...")
    reference_histograms = load_reference_histograms(REFERENCE_DIR)
    print(f"Loaded references for: {list(reference_histograms.keys())}")
    
    # Initialize confusion matrix and FN counter
    # Confusion matrix: rows are predicted classes, columns are GT classes + FP column
    gt_labels_with_fp = CLASS_LABELS + [FP_LABEL]
    conf_matrix = {pred: {gt: 0 for gt in gt_labels_with_fp} for pred in CLASS_LABELS}
    
    # False Negatives tracked separately (not in confusion matrix)
    false_negatives = {label: 0 for label in CLASS_LABELS}
    
    # Count total frames for ETA
    total_frames = 0
    for test in TEST_VIDEOS:
        cap_tmp = cv2.VideoCapture(test["video"])
        total_frames += int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_tmp.release()
    
    processed_frames = 0
    start_time = time.time()
    
    print(f"\nTotal frames to process: {total_frames}")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Effective frames: ~{total_frames // FRAME_SKIP}")
    print("\n" + "=" * 80)
    
    # Process each video
    for video_idx, test in enumerate(TEST_VIDEOS, 1):
        video_path = test["video"]
        json_path = test["json"]
        
        print(f"\n[VIDEO {video_idx}] {os.path.basename(video_path)}")
        print("-" * 80)
        
        # Load GT annotations
        gt_frames = load_coco_annotations(json_path)
        print(f"Loaded {len(gt_frames)} frames with annotations")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            # Skip frames according to FRAME_SKIP
            if frame_id % FRAME_SKIP != 0:
                continue
            
            processed_frames += 1
            
            # Get ground truth for this frame
            gt_boxes = gt_frames.get(frame_id, [])
            
            # Detect people
            detections = detect_people(frame, clf, hog)  # Returns [[x1,y1,x2,y2], ...]
            
            # Classify each detection (without printing)
            det_classes = []
            for det_box in detections:
                x1, y1, x2, y2 = det_box
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    det_classes.append("Unidentified soldier")
                    continue
                pred_label = classify_detection(roi, reference_histograms)
                det_classes.append(pred_label)
            
            # Match detections to GT
            matched_pairs, unmatched_dets, unmatched_gts = match_detections_to_gt(
                detections, gt_boxes, IOU_THRESHOLD
            )
            
            # Update confusion matrix for matched pairs (True Positives)
            for det_idx, gt_idx in matched_pairs:
                pred_label = det_classes[det_idx]
                gt_label = gt_boxes[gt_idx]["label"]
                conf_matrix[pred_label][gt_label] += 1
            
            # Handle False Positives (detections with no matching GT)
            # Add to confusion matrix under FP_LABEL column
            for det_idx in unmatched_dets:
                pred_label = det_classes[det_idx]
                conf_matrix[pred_label][FP_LABEL] += 1
            
            # Count False Negatives (unmatched GT)
            for gt_idx in unmatched_gts:
                gt_label = gt_boxes[gt_idx]["label"]
                false_negatives[gt_label] += 1
            
            # Progress print every 5 frames
            if processed_frames % 5 == 0 or processed_frames == 1:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed if elapsed > 0 else 0
                remaining_frames = (total_frames // FRAME_SKIP) - processed_frames
                eta_seconds = remaining_frames / fps if fps > 0 else 0
                
                # Calculate current counts
                total_fp = sum(conf_matrix[pred][FP_LABEL] for pred in CLASS_LABELS)
                total_fn = sum(false_negatives.values())
                
                # Count ALL predictions (TP + FP) per class
                class_counts = {}
                for pred_label in CLASS_LABELS:
                    tp_count = sum(conf_matrix[pred_label][gt] for gt in CLASS_LABELS)
                    fp_count = conf_matrix[pred_label][FP_LABEL]
                    total_pred = tp_count + fp_count
                    if total_pred > 0:
                        class_counts[pred_label] = total_pred
                
                # Build class count string
                class_str = " | ".join([f"{label}: {count}" for label, count in class_counts.items()])
                
                print(f"\r[Progress] {processed_frames}/{total_frames // FRAME_SKIP} | "
                      f"Elapsed: {int(elapsed)}s | ETA: {int(eta_seconds)}s | FPS: {fps:.2f} | "
                      f"FN: {total_fn} | FP: {total_fp}")
                if class_str:
                    print(f"[Classes] {class_str}")
        
        cap.release()
        print()  # New line after progress
    
    # ============ FINAL RESULTS ============
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Combined for all videos)")
    print("=" * 80)
    
    # Print confusion matrix
    print("\nConfusion Matrix (Predicted vs Ground Truth + False Positives):")
    print("-" * 120)
    
    # Header
    header = "Predicted \\ GT".ljust(25)
    for gt_label in CLASS_LABELS:
        header += gt_label[:15].ljust(17)
    header += FP_LABEL.ljust(17)
    print(header)
    print("-" * 120)
    
    # Rows - actual classes
    for pred_label in CLASS_LABELS:
        row = pred_label[:23].ljust(25)
        for gt_label in CLASS_LABELS:
            count = conf_matrix[pred_label][gt_label]
            row += str(count).ljust(17)
        # Add FP column
        fp_count = conf_matrix[pred_label][FP_LABEL]
        row += str(fp_count).ljust(17)
        print(row)
    
    # Print class totals from confusion matrix
    print("\n" + "=" * 120)
    print("Detection Summary per Predicted Class:")
    print("-" * 120)
    for pred_label in CLASS_LABELS:
        tp_count = sum(conf_matrix[pred_label][gt] for gt in CLASS_LABELS)
        fp_count = conf_matrix[pred_label][FP_LABEL]
        total = tp_count + fp_count
        if total > 0:
            print(f"  {pred_label}: {total} detections (TP: {tp_count}, FP: {fp_count})")
    
    # Print False Negatives
    print("\n" + "=" * 120)
    print("False Negatives (Ground Truth not detected):")
    print("-" * 120)
    total_fn = 0
    for label in CLASS_LABELS:
        fn_count = false_negatives[label]
        if fn_count > 0:
            print(f"  {label}: {fn_count}")
            total_fn += fn_count
    print(f"\nTotal False Negatives: {total_fn}")
    
    # Summary stats
    print("\n" + "=" * 120)
    print("Summary Statistics:")
    print("-" * 120)
    total_matches = sum(sum(conf_matrix[pred][gt] for gt in CLASS_LABELS) for pred in CLASS_LABELS)
    total_fp = sum(conf_matrix[pred][FP_LABEL] for pred in CLASS_LABELS)
    total_detections = total_matches + total_fp
    total_gt = total_matches + total_fn
    detection_rate = (total_matches / total_gt * 100) if total_gt > 0 else 0
    precision = (total_matches / total_detections * 100) if total_detections > 0 else 0
    
    print(f"  Total Ground Truth objects: {total_gt}")
    print(f"  Total Detections made: {total_detections}")
    print(f"  True Positives (matched): {total_matches}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Detection Rate (Recall): {detection_rate:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    
    elapsed_total = time.time() - start_time
    print(f"\nTotal processing time: {int(elapsed_total)}s ({elapsed_total / 60:.2f} min)")
    print("=" * 120)
    
    # ============ VISUALIZATION ============
    print("\nGenerating confusion matrix visualization...")
    
    # Build confusion matrix as numpy array
    all_labels = CLASS_LABELS + [FP_LABEL]
    matrix = np.zeros((len(CLASS_LABELS), len(all_labels)))
    
    for i, pred_label in enumerate(CLASS_LABELS):
        for j, gt_label in enumerate(all_labels):
            matrix[i, j] = conf_matrix[pred_label][gt_label]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Full confusion matrix with FP column
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=all_labels, yticklabels=CLASS_LABELS,
                cbar_kws={'label': 'Count'}, ax=ax1, square=False)
    ax1.set_xlabel('Ground Truth (Actual Class)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Predicted (Model Output)', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_title('Confusion Matrix\n(Rows = What Model Predicted, Columns = Actual GT)', 
                  fontsize=14, fontweight='bold', pad=15)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=10)
    
    # Plot 2: Statistics summary
    ax2.axis('off')
    stats_text = f"""
    FULL EVALUATION RESULTS
    {'='*50}
    
    Total Frames Processed: {processed_frames}
    Processing Time: {int(elapsed_total)}s ({elapsed_total/60:.2f} min)
    FPS: {processed_frames/elapsed_total:.2f}
    
    DETECTION METRICS
    {'='*50}
    Total Ground Truth Objects: {total_gt}
    Total Detections Made: {total_detections}
    
    True Positives (TP): {total_matches}
    False Positives (FP): {total_fp}
    False Negatives (FN): {total_fn}
    
    Detection Rate (Recall): {detection_rate:.2f}%
    Precision: {precision:.2f}%
    F1-Score: {(2 * precision * detection_rate / (precision + detection_rate) if (precision + detection_rate) > 0 else 0):.2f}%
    
    PER-CLASS DETECTIONS
    {'='*50}
    """
    
    for pred_label in CLASS_LABELS:
        tp_count = sum(conf_matrix[pred_label][gt] for gt in CLASS_LABELS)
        fp_count = conf_matrix[pred_label][FP_LABEL]
        total = tp_count + fp_count
        if total > 0:
            stats_text += f"\n{pred_label}: {total} (TP: {tp_count}, FP: {fp_count})"
    
    stats_text += f"\n\nFALSE NEGATIVES (GT NOT DETECTED)\n{'='*50}"
    for label in CLASS_LABELS:
        fn_count = false_negatives[label]
        if fn_count > 0:
            stats_text += f"\n{label}: {fn_count}"
    
    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = "Output/confusion_matrix_full.png"
    os.makedirs("Output", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    
    # Show figure
    plt.show()
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
