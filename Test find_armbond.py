import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define multiple video/annotation pairs
DATASET = [
    {
        "video": r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4",
        "json": r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\Testing\3mili 2 onde 1 god.json"
    },
    {
        "video": r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\3 mili 2 onde 1 god.MP4",
        "json": r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json"
    }
    # Add more video/json pairs here as needed
]

OUTPUT_DIR = r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\Output"

def load_coco_annotations(json_path):
    """Load COCO JSON and organize annotations by frame"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image_id to frame info
    image_id_to_frame = {}
    for img in data['images']:
        image_id_to_frame[img['id']] = img
    
    # Organize annotations by image_id
    frame_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in frame_annotations:
            frame_annotations[image_id] = []
        frame_annotations[image_id].append(ann)
    
    return frame_annotations, image_id_to_frame, data

def get_ground_truth_label(annotation):
    """Extract ground truth label from COCO annotation attributes"""
    attrs = annotation.get('attributes', {})
    
    # Check for HVT first
    if attrs.get('Good HVT', False):
        return 'Good HVT'
    elif attrs.get('Bad HVT', False):
        return 'Bad HVT'
    # Then check for regular soldiers
    elif attrs.get('Military good', False):
        return 'Good soldier'
    elif attrs.get('Military bad', False):
        return 'Bad soldier'
    elif attrs.get('Military', False):
        return 'Soldier'
    elif attrs.get('Civilian', False):
        return 'Civilian'
    elif attrs.get('Unknown person', False):
        return 'Unknown'
    else:
        return 'Soldier'  # Default

def normalize_prediction(classification):
    """Normalize prediction to match ground truth labels"""
    # Map various prediction strings to standard labels
    if 'Good soldier (HVT)' in classification or 'Good HVT' in classification:
        return 'Good HVT'
    elif 'Bad soldier (HVT)' in classification or 'Bad HVT' in classification:
        return 'Bad HVT'
    elif 'Good soldier' in classification:
        return 'Good soldier'
    elif 'Bad soldier' in classification:
        return 'Bad soldier'
    else:
        return 'Soldier'

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Generate and save confusion matrix as PNG"""
    # Define class order
    labels = ['Good soldier', 'Bad soldier', 'Good HVT', 'Bad HVT', 'Soldier']
    
    # Filter to only include labels that appear in the data
    unique_labels = sorted(set(y_true + y_pred))
    labels = [l for l in labels if l in unique_labels]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Soldier Classification', fontsize=16, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_path}")
    
    # Calculate and print metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    # Calculate overall accuracy
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"Total Samples: {len(y_true)}")
    
    plt.close()

def blob_analysis(img, morph_kernel=(3,3), morph_iters=1, min_pixels=1, rel_area_multiplier=0.004, max_components=2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV ranges - Tune these values directly
    # Red ranges (wraps around H=0/180)
    RED_LOWER1 = [5, 160, 75]      # [H, S, V]
    RED_UPPER1 = [10, 255, 255]
    RED_LOWER2 = [170, 160, 75]
    RED_UPPER2 = [180, 255, 255]
    
    # Blue range
    BLUE_LOWER = [100, 100, 50]
    BLUE_UPPER = [130, 255, 255]

    # Build masks using cv2.inRange directly
    mask_red1 = cv2.inRange(hsv, np.array(RED_LOWER1, dtype=np.uint8), np.array(RED_UPPER1, dtype=np.uint8))
    mask_red2 = cv2.inRange(hsv, np.array(RED_LOWER2, dtype=np.uint8), np.array(RED_UPPER2, dtype=np.uint8))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_blue = cv2.inRange(hsv, np.array(BLUE_LOWER, dtype=np.uint8), np.array(BLUE_UPPER, dtype=np.uint8))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=morph_iters)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=morph_iters)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)

    # Find connected components and filter small ones by area (absolute + relative)
    h, w = hsv.shape[:2]
    rel_min = int(h * w * rel_area_multiplier)
    min_area = max(int(min_pixels), rel_min)

    def find_components(mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                ww = int(stats[i, cv2.CC_STAT_WIDTH])
                hh = int(stats[i, cv2.CC_STAT_HEIGHT])
                boxes.append((x, y, ww, hh, area))
        return boxes

    red_boxes = find_components(mask_red)
    blue_boxes = find_components(mask_blue)

    # Keep only the largest `max_components` per color
    if len(red_boxes) > max_components:
        red_boxes = sorted(red_boxes, key=lambda b: b[4], reverse=True)[:max_components]
    if len(blue_boxes) > max_components:
        blue_boxes = sorted(blue_boxes, key=lambda b: b[4], reverse=True)[:max_components]

    # Annotate the image
    out = img.copy()
    for (x, y, ww, hh, area) in red_boxes:
        cv2.rectangle(out, (x, y), (x+ww, y+hh), (0,0,255), 2)
    for (x, y, ww, hh, area) in blue_boxes:
        cv2.rectangle(out, (x, y), (x+ww, y+hh), (255,0,0), 2)

    return out, mask_red, mask_blue, red_boxes, blue_boxes

def classify_target(red_boxes, blue_boxes):
    num_red = len(red_boxes)
    num_blue = len(blue_boxes)
    
    # No boxes found
    if num_red == 0 and num_blue == 0:
        return "Soldier", None, False
    
    # Normal cases - single color only
    if num_blue > 0 and num_red == 0:
        if num_blue == 1:
            return "Good soldier", "good", False
        elif num_blue == 2:
            return "Good soldier (HVT)", "good", True
    
    if num_red > 0 and num_blue == 0:
        if num_red == 1:
            return "Bad soldier", "bad", False
        elif num_red == 2:
            return "Bad soldier (HVT)", "bad", True
    
    # Mixed cases - both colors present
    # Case: 2 blue + 1 red
    if num_blue == 2 and num_red == 1:
        return "Good soldier (HVT)", "good", True
    
    # Case: 2 red + 1 blue
    if num_red == 2 and num_blue == 1:
        return "Bad soldier (HVT)", "bad", True
    
    # Case: 1 red + 1 blue -> compare areas
    if num_red == 1 and num_blue == 1:
        # boxes format: (x, y, w, h, area)
        red_area = red_boxes[0][4]
        blue_area = blue_boxes[0][4]
        
        if blue_area > red_area:
            return "Good soldier (area-based)", "good", False
        else:
            return "Bad soldier (area-based)", "bad", False
    
    if num_red == 2 and num_blue == 2:
        # Compare largest areas
        red_area = max(box[4] for box in red_boxes)
        blue_area = max(box[4] for box in blue_boxes)
        
        if blue_area > red_area:
            return "Good soldier (HVT area-based)", "good", True
        else:
            return "Bad soldier (HVT area-based)", "bad", True
    
    # Fallback for any other combinations
    return "Uncertain classification", None, False

def crop_to_bbox(img, bbox):
    """Crop image to bounding box [x, y, width, height]"""
    x, y, w, h = [int(v) for v in bbox]
    # Ensure coordinates are within image bounds
    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return img[y:y+h, x:x+w]

def process_bbox_region(bbox_img):
    """Process a bounding box region: crop, blur, and analyze"""
    if bbox_img is None or bbox_img.size == 0:
        return None, [], []
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(bbox_img, (13, 13), 0)
    
    # Run blob analysis
    annotated, _, _, red_boxes, blue_boxes = blob_analysis(blurred, morph_kernel=(3, 3), morph_iters=1)
    
    return annotated, red_boxes, blue_boxes

def process_video(video_path, json_path, scale=0.5, show_video=True):
    """Process video frame by frame, classify soldiers in COCO bounding boxes, and display results.
    
    Args:
        video_path: Path to video file
        json_path: Path to COCO JSON annotations
        scale: Scale factor for display (0.5 = 50%, 1.0 = 100%)
        show_video: Whether to show video window (set False for faster processing)
    """
    print(f"Loading annotations from: {json_path}")
    frame_annotations, image_id_to_frame, coco_data = load_coco_annotations(json_path)
    print(f"Loaded {len(frame_annotations)} frames with annotations")
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {w}x{h}, {fps} fps, {total_frames} frames")
    
    # Calculate display dimensions
    display_w = int(w * scale)
    display_h = int(h * scale)
    
    if show_video:
        cv2.namedWindow("Video - Press SPACE/any key to advance, Q/ESC to quit", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video - Press SPACE/any key to advance, Q/ESC to quit", display_w, display_h)
    
    frame_idx = 0
    processed_count = 0
    
    # Storage for confusion matrix
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels
    
    # Try different offsets for frame to image_id mapping
    offsets_to_try = [0, 1, 3720]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\nEnd of video reached. Processed {processed_count} frames with annotations.")
            break
        
        display_frame = frame.copy()
        
        # Try to find annotations for this frame using different image_id mappings
        image_id = None
        annotations = None
        for offset in offsets_to_try:
            test_id = frame_idx + offset
            if test_id in frame_annotations:
                image_id = test_id
                annotations = frame_annotations[test_id]
                break
        
        if annotations:
            # Process each bounding box in this frame
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w_box, h_box = [int(v) for v in bbox]
                
                # Extract the region within the bounding box
                bbox_img = crop_to_bbox(frame, bbox)
                
                if bbox_img is not None and bbox_img.size > 0:
                    # Process the bbox region
                    annotated_bbox, red_boxes, blue_boxes = process_bbox_region(bbox_img)
                    
                    if annotated_bbox is not None:
                        # Classify the target
                        classification, target_type, is_hvt = classify_target(red_boxes, blue_boxes)
                        
                        # Get ground truth label
                        ground_truth = get_ground_truth_label(ann)
                        predicted = normalize_prediction(classification)
                        
                        # Store for confusion matrix
                        y_true.append(ground_truth)
                        y_pred.append(predicted)
                        
                        # Choose color based on classification
                        if target_type == "good":
                            box_color = (0, 255, 0)  # Green for good soldier
                        elif target_type == "bad":
                            box_color = (0, 0, 255)  # Red for bad soldier
                        else:
                            box_color = (255, 255, 0)  # Cyan for uncertain
                        
                        # Draw bounding box on display frame
                        thickness = 3 if is_hvt else 2
                        cv2.rectangle(display_frame, (x, y), (x + w_box, y + h_box), box_color, thickness)
                        
                        # Add classification text
                        label = classification
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2
                        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
                        
                        # Draw text background
                        cv2.rectangle(display_frame, (x, y - text_h - 10), (x + text_w + 10, y), box_color, -1)
                        # Draw text
                        cv2.putText(display_frame, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), text_thickness)
                        
                        if show_video:
                            print(f"Frame {frame_idx}: GT={ground_truth}, Pred={predicted}")
            
            processed_count += 1
        
        if show_video:
            # Add frame counter
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize for display
            if scale != 1.0:
                display_frame = cv2.resize(display_frame, (display_w, display_h))
            
            cv2.imshow("Video - Press SPACE/any key to advance, Q/ESC to quit", display_frame)
            
            # Wait for key press (0 = wait indefinitely)
            k = cv2.waitKey(0) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):  # ESC or Q
                print("\nQuitting early...")
                break
        else:
            # Auto-advance without display
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}...")
        
        frame_idx += 1
    
    cap.release()
    if show_video:
        cv2.destroyAllWindows()
    print(f"\nProcessing complete! Processed {processed_count} frames with annotations.")
    
    return y_true, y_pred


if __name__ == "__main__":
    # Process all videos and collect predictions
    all_y_true = []
    all_y_pred = []
    
    print(f"Processing {len(DATASET)} video(s)...")
    print("="*60)
    
    for idx, data in enumerate(DATASET):
        video_path = data["video"]
        json_path = data["json"]
        
        print(f"\n[Dataset {idx+1}/{len(DATASET)}]")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"JSON: {os.path.basename(json_path)}")
        print("-"*60)
        
        y_true, y_pred = process_video(video_path, json_path, scale=0.5, show_video=False)
        
        if y_true and y_pred:
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            print(f"Collected {len(y_true)} predictions from this video")
        else:
            print(f"No predictions collected from this video")
    
    print("\n" + "="*60)
    print(f"TOTAL PREDICTIONS COLLECTED: {len(all_y_true)}")
    print("="*60)
    
    if all_y_true and all_y_pred:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generate confusion matrix
        output_path = os.path.join(OUTPUT_DIR, "confusion_matrix_combined.png")
        plot_confusion_matrix(all_y_true, all_y_pred, output_path)
        
        print(f"\nCombined confusion matrix saved to: {output_path}")
    else:
        print("No predictions collected from any video!")
