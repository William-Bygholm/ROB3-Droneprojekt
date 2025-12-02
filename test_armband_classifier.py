"""
Test armband classification against ground truth annotations.
Generates precision-recall curves and performance metrics.
"""

import cv2
import numpy as np
import json
import os
from find_armbod import blob_analysis, classify_target, crop_image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns

# Configuration
TEST_DATA = {
    'video': r"ProjektVideoer/2 militær med blå bånd .MP4",
    'json': r"Validation/2 mili med blå bond.json"
}

# Class mapping from COCO attributes to our classification
CLASS_MAPPING = {
    'Military good': 'good',
    'Military bad': 'bad',
    'Good HVT': 'good_hvt',
    'Bad HVT': 'bad_hvt',
    'Civilian': 'civilian',
    'Unknown person': 'unknown'
}

def load_ground_truth(json_path):
    """Load COCO annotations and organize by frame"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Organize by image_id
    frame_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in frame_annotations:
            frame_annotations[image_id] = []
        
        # Extract class from attributes
        attrs = ann.get('attributes', {})
        gt_class = None
        is_hvt = False
        
        for class_name, is_active in attrs.items():
            if is_active and class_name in CLASS_MAPPING:
                gt_class = CLASS_MAPPING[class_name]
                if 'HVT' in class_name:
                    is_hvt = True
                break
        
        frame_annotations[image_id].append({
            'bbox': ann['bbox'],  # [x, y, width, height]
            'class': gt_class,
            'is_hvt': is_hvt,
            'track_id': attrs.get('track_id', -1)
        })
    
    return frame_annotations

def extract_person_roi(frame, bbox, padding=0.1):
    """Extract ROI from frame with padding"""
    h, w = frame.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    
    # Add padding
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)
    
    return frame[y1:y2, x1:x2]

def classify_person_armband(roi):
    """
    Classify a person ROI using armband detection.
    Returns: (predicted_class, confidence, details)
    """
    if roi.shape[0] < 20 or roi.shape[1] < 20:
        return None, 0.0, "ROI too small"
    
    # Crop to focus on upper body (where armbands are)
    cropped = crop_image(roi)
    
    # Apply blob analysis
    _, mask_red, mask_blue, red_boxes, blue_boxes = blob_analysis(
        cropped,
        morph_kernel=(3, 3),
        morph_iters=1,
        min_pixels=10,
        rel_area_multiplier=0.002,
        max_components=2
    )
    
    # Classify based on detected armbands
    classification, target_type, is_hvt = classify_target(red_boxes, blue_boxes)
    
    # Convert to our class format
    if target_type == "good":
        pred_class = "good_hvt" if is_hvt else "good"
    elif target_type == "bad":
        pred_class = "bad_hvt" if is_hvt else "bad"
    else:
        pred_class = None
    
    # Calculate confidence based on detection quality
    total_boxes = len(red_boxes) + len(blue_boxes)
    if total_boxes == 0:
        confidence = 0.0
    elif total_boxes == 1:
        confidence = 0.6  # Single armband - moderate confidence
    elif total_boxes == 2 and (len(red_boxes) == 2 or len(blue_boxes) == 2):
        confidence = 0.9  # Two same color - high confidence (HVT)
    else:
        confidence = 0.4  # Mixed signals - low confidence
    
    details = {
        'classification': classification,
        'red_boxes': len(red_boxes),
        'blue_boxes': len(blue_boxes),
        'is_hvt': is_hvt
    }
    
    return pred_class, confidence, details

def test_video(video_path, ground_truth, sample_every_n=1):
    """
    Test armband classifier on video with ground truth.
    Returns predictions and ground truth for evaluation.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Results storage
    predictions = []  # List of (pred_class, confidence, gt_class)
    
    print(f"Testing on {total_frames} frames...")
    print("="*60)
    
    frames_tested = 0
    people_tested = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_every_n != 0:
            frame_idx += 1
            continue
        
        # Find ground truth for this frame
        gt_annotations = None
        for possible_id in [frame_idx, frame_idx + 1, frame_idx - 1]:
            if possible_id in ground_truth:
                gt_annotations = ground_truth[possible_id]
                break
        
        if gt_annotations:
            frames_tested += 1
            
            for ann in gt_annotations:
                people_tested += 1
                
                # Extract person ROI
                roi = extract_person_roi(frame, ann['bbox'], padding=0.1)
                
                # Run classification
                pred_class, confidence, details = classify_person_armband(roi)
                
                # Store result
                gt_class = ann['class']
                predictions.append({
                    'frame': frame_idx,
                    'track_id': ann['track_id'],
                    'predicted': pred_class,
                    'confidence': confidence,
                    'ground_truth': gt_class,
                    'details': details
                })
                
                if people_tested % 50 == 0:
                    print(f"Tested {people_tested} people from {frames_tested} frames...")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\nTesting complete!")
    print(f"Frames tested: {frames_tested}")
    print(f"People tested: {people_tested}")
    print("="*60)
    
    return predictions

def evaluate_predictions(predictions, output_dir="Output"):
    """
    Evaluate predictions and generate metrics, confusion matrix, and PR curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out cases where we have both prediction and ground truth
    valid_predictions = [p for p in predictions if p['predicted'] is not None and p['ground_truth'] is not None]
    
    if len(valid_predictions) == 0:
        print("ERROR: No valid predictions to evaluate!")
        return
    
    print(f"\nEvaluating {len(valid_predictions)} valid predictions...")
    
    # Extract data
    y_true = [p['ground_truth'] for p in valid_predictions]
    y_pred = [p['predicted'] for p in valid_predictions]
    confidences = [p['confidence'] for p in valid_predictions]
    
    # Get unique classes
    classes = sorted(list(set(y_true + y_pred)))
    
    print(f"\nClasses found: {classes}")
    
    # Simple accuracy
    correct = sum(1 for p in valid_predictions if p['predicted'] == p['ground_truth'])
    accuracy = correct / len(valid_predictions)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{len(valid_predictions)})")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Armband Classification')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    print(f"Saved: {output_dir}/confusion_matrix.png")
    plt.close()
    
    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS METRICS:")
    print("="*60)
    
    for cls in classes:
        tp = sum(1 for p in valid_predictions if p['ground_truth'] == cls and p['predicted'] == cls)
        fp = sum(1 for p in valid_predictions if p['ground_truth'] != cls and p['predicted'] == cls)
        fn = sum(1 for p in valid_predictions if p['ground_truth'] == cls and p['predicted'] != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{cls}:")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall:    {recall:.2%}")
        print(f"  F1-Score:  {f1:.2%}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
    
    # Binary classification for good vs bad (ignoring HVT distinction)
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION (Good vs Bad):")
    print("="*60)
    
    y_true_binary = ['good' if 'good' in gt else 'bad' if 'bad' in gt else 'other' for gt in y_true]
    y_pred_binary = ['good' if 'good' in pred else 'bad' if 'bad' in pred else 'other' for pred in y_pred]
    
    # Filter out 'other'
    binary_pairs = [(gt, pred, conf) for gt, pred, conf in zip(y_true_binary, y_pred_binary, confidences) 
                    if gt != 'other' and pred != 'other']
    
    if len(binary_pairs) > 0:
        y_true_bin = [p[0] for p in binary_pairs]
        y_pred_bin = [p[1] for p in binary_pairs]
        conf_bin = [p[2] for p in binary_pairs]
        
        correct_bin = sum(1 for gt, pred in zip(y_true_bin, y_pred_bin) if gt == pred)
        accuracy_bin = correct_bin / len(y_true_bin)
        print(f"Binary Accuracy: {accuracy_bin:.2%} ({correct_bin}/{len(y_true_bin)})")
        
        # Precision-Recall curve for "good" class
        y_true_numeric = [1 if gt == 'good' else 0 for gt in y_true_bin]
        y_score = [conf if pred == 'good' else (1 - conf) for pred, conf in zip(y_pred_bin, conf_bin)]
        
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true_numeric, y_score)
        avg_precision = average_precision_score(y_true_numeric, y_score)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall_curve, precision_curve, linewidth=2, label=f'AP = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Good Soldier Detection)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=150)
        print(f"\nAverage Precision (Good): {avg_precision:.3f}")
        print(f"Saved: {output_dir}/precision_recall_curve.png")
        plt.close()
    
    # Save detailed results to CSV
    import csv
    csv_path = os.path.join(output_dir, 'test_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'track_id', 'ground_truth', 'predicted', 
                                                'confidence', 'correct', 'red_boxes', 'blue_boxes'])
        writer.writeheader()
        for p in valid_predictions:
            writer.writerow({
                'frame': p['frame'],
                'track_id': p['track_id'],
                'ground_truth': p['ground_truth'],
                'predicted': p['predicted'],
                'confidence': p['confidence'],
                'correct': p['ground_truth'] == p['predicted'],
                'red_boxes': p['details']['red_boxes'],
                'blue_boxes': p['details']['blue_boxes']
            })
    print(f"Saved: {csv_path}")

def main():
    print("="*60)
    print("ARMBAND CLASSIFIER EVALUATION")
    print("="*60)
    
    # Check files exist
    if not os.path.exists(TEST_DATA['video']):
        print(f"ERROR: Video not found: {TEST_DATA['video']}")
        return
    if not os.path.exists(TEST_DATA['json']):
        print(f"ERROR: JSON not found: {TEST_DATA['json']}")
        return
    
    # Load ground truth
    print(f"\nLoading ground truth from: {TEST_DATA['json']}")
    ground_truth = load_ground_truth(TEST_DATA['json'])
    print(f"Loaded {len(ground_truth)} annotated frames")
    
    # Test on video
    print(f"\nTesting video: {TEST_DATA['video']}")
    predictions = test_video(TEST_DATA['video'], ground_truth, sample_every_n=1)
    
    # Evaluate
    evaluate_predictions(predictions)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
