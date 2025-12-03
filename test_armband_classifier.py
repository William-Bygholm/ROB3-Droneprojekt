"""
Test armband classification against ground truth annotations.
Generates confusion matrix and performance metrics.
"""

import cv2
import numpy as np
import json
import os
from find_armbod import blob_analysis, classify_target, crop_image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuration
TEST_DATA = [
    {
        'video': r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4",
        'json': r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\Validation\2 mili med blå bond.json"
    }
]

# Class mapping from COCO attributes to our classification
CLASS_MAPPING = {
    'Good military': 'good',
    'Bad military': 'bad',
    'Good military (HVT)': 'good_hvt',
    'Bad military (HVT)': 'bad_hvt',
    'Civilian': 'civilian',
    'Military': 'military',
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

def classify_person_armband(roi, display_debug=False, display_height=300, step_mode=False):
    """
    Classify a person ROI using armband detection.
    Returns: (predicted_class, confidence, details)
    """
    if roi.shape[0] < 20 or roi.shape[1] < 20:
        return None, 0.0, "ROI too small"
    
    # Crop to focus on upper body (where armbands are)
    cropped = crop_image(roi)
    
    # Apply blob analysis
    annotated, mask_red, mask_blue, red_boxes, blue_boxes = blob_analysis(
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
    
    # Display blob analysis if requested
    if display_debug:
        # Create visualization
        debug_img = annotated.copy()
        
        # Add text overlay
        cv2.putText(debug_img, f"Pred: {pred_class}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Conf: {confidence:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"R:{len(red_boxes)} B:{len(blue_boxes)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show masks side by side
        mask_red_colored = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
        mask_blue_colored = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)
        
        # Resize for display (configurable height)
        scale = display_height / debug_img.shape[0] if debug_img.shape[0] > 0 else 1
        debug_resized = cv2.resize(debug_img, None, fx=scale, fy=scale)
        mask_red_resized = cv2.resize(mask_red_colored, None, fx=scale, fy=scale)
        mask_blue_resized = cv2.resize(mask_blue_colored, None, fx=scale, fy=scale)
        
        # Combine horizontally
        combined = np.hstack([debug_resized, mask_red_resized, mask_blue_resized])
        
        # Create resizable window and apply size
        win_name = "Blob Analysis: Image | Red Mask | Blue Mask"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        # Try to set window size proportional to combined image
        desired_width = int(combined.shape[1])
        desired_height = int(combined.shape[0])
        cv2.resizeWindow(win_name, desired_width, desired_height)
        cv2.imshow(win_name, combined)
        # Step-through mode waits for keypress
        if step_mode:
            key = cv2.waitKey(0)
            # Allow quit
            if key in (ord('q'), 27):  # 'q' or ESC
                # Signal quit by raising KeyboardInterrupt; caller handles
                raise KeyboardInterrupt("User aborted visualization")
        else:
            cv2.waitKey(1)
    
    details = {
        'classification': classification,
        'red_boxes': len(red_boxes),
        'blue_boxes': len(blue_boxes),
        'is_hvt': is_hvt
    }
    
    return pred_class, confidence, details

def test_video(video_path, ground_truth, sample_every_n=1, display_debug=False, display_height=300, step_mode=False):
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
                try:
                    pred_class, confidence, details = classify_person_armband(
                        roi,
                        display_debug=display_debug,
                        display_height=display_height,
                        step_mode=step_mode,
                    )
                except KeyboardInterrupt:
                    # User chose to abort visualization; break out cleanly
                    cap.release()
                    return predictions
                
                # Store result
                gt_class = ann['class']
                
                # Print classification decision
                #print(f"\n  Frame {frame_idx}, Track {ann['track_id']}:")
                #print(f"    Ground Truth: {gt_class}")
                #print(f"    Predicted: {pred_class} (confidence: {confidence:.2f})")
                #print(f"    Details: {details['classification']}")
                #print(f"    Red boxes: {details['red_boxes']}, Blue boxes: {details['blue_boxes']}, HVT: {details['is_hvt']}")
                #print(f"    Match: {'✓' if pred_class == gt_class else '✗'}")
                
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
    binary_pairs = [(gt, pred) for gt, pred in zip(y_true_binary, y_pred_binary) 
                    if gt != 'other' and pred != 'other']
    
    if len(binary_pairs) > 0:
        y_true_bin = [p[0] for p in binary_pairs]
        y_pred_bin = [p[1] for p in binary_pairs]
        
        correct_bin = sum(1 for gt, pred in zip(y_true_bin, y_pred_bin) if gt == pred)
        accuracy_bin = correct_bin / len(y_true_bin)
        print(f"Binary Accuracy: {accuracy_bin:.2%} ({correct_bin}/{len(y_true_bin)})")

def main():
    print("="*60)
    print("ARMBAND CLASSIFIER EVALUATION")
    print("="*60)
    
    # Ask user if they want to display debug visualization
    print("\nDisplay blob analysis visualization? (y/n): ", end='')
    display_debug = input().strip().lower() == 'y'
    
    # Collect predictions from all videos
    all_predictions = []
    
    for idx, dataset in enumerate(TEST_DATA, 1):
        video_path = dataset['video']
        json_path = dataset['json']
        
        print(f"\n{'='*60}")
        print(f"Processing dataset {idx}/{len(TEST_DATA)}")
        print(f"Video: {video_path}")
        print(f"JSON: {json_path}")
        print('='*60)
        
        # Check files exist
        if not os.path.exists(video_path):
            print("\nDisplay blob analysis visualization? (y/N): ", end='') 
            display_debug = input().strip().lower() == 'y'
            step_mode = False
            display_height = 750
            if display_debug:
                try:
                    step_choice = input("Enable step-through mode? Wait for key per person (y/N): ").strip().lower()
                    step_mode = step_choice == 'y'
                    h_input = input("Desired display height in pixels (default 300): ").strip()
                    if h_input:
                        display_height = max(100, int(h_input))
                except Exception:
                    print("Using default display settings (height=300, step_mode=False)")
            continue
        if not os.path.exists(json_path):
            print(f"WARNING: JSON not found, skipping: {json_path}")
            continue
        
        # Load ground truth
        print(f"\nLoading ground truth from: {json_path}")
        ground_truth = load_ground_truth(json_path)
        print(f"Loaded {len(ground_truth)} annotated frames")
        
        # Test on video
        print(f"\nTesting video: {video_path}")
        predictions = test_video(video_path, ground_truth, sample_every_n=1, display_debug=display_debug)
        
        all_predictions.extend(predictions)
        print(f"Collected {len(predictions)} predictions from this video")
    
    # Close any visualization windows
    cv2.destroyAllWindows()
    
    # Evaluate combined results
    print(f"\n{'='*60}")
    print(f"COMBINED EVALUATION")
    print(f"Total predictions from all videos: {len(all_predictions)}")
    print('='*60)
    
    evaluate_predictions(all_predictions)
    
    print(f"\nTesting video: {video_path}")
    predictions = test_video(
    video_path,
    ground_truth,
    sample_every_n=1,
    display_debug=display_debug,
    display_height=display_height,
    step_mode=step_mode,
    )
    print("="*60)

if __name__ == "__main__":
    main()
