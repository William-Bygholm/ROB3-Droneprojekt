import os
import cv2
import numpy as np
import json

def load_coco_annotations(json_path):
    """Load COCO JSON and organize annotations by frame"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image_id to annotations
    frame_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in frame_annotations:
            frame_annotations[image_id] = []
        frame_annotations[image_id].append(ann)
    
    # Create mapping from image_id to frame number (if available)
    image_info = {img['id']: img for img in data['images']}
    
    print(f"Found {len(image_info)} images with {len(frame_annotations)} annotated frames")
    if image_info:
        sample_ids = list(image_info.keys())[:5]
        print(f"Sample image IDs: {sample_ids}")
    
    return frame_annotations, image_info

def get_video_path(video_file):

    if os.path.isabs(video_file):
        video_path = video_file
    else:
        video_path = os.path.join(os.path.dirname(__file__), video_file)
    
    if not os.path.isfile(video_path):
        raise ValueError(f"The video file '{video_path}' does not exist.")
    
    return video_path

def color_mask(hsv, lower, upper):
    """Handle hue wrap-around when creating an inRange mask."""
    lh, ls, lv = int(lower[0]), int(lower[1]), int(lower[2])
    uh, us, uv = int(upper[0]), int(upper[1]), int(upper[2])
    if lh <= uh:
        return cv2.inRange(hsv, np.array([lh, ls, lv], np.uint8), np.array([uh, us, uv], np.uint8))
    # wrap around 180 -> union of two ranges
    m1 = cv2.inRange(hsv, np.array([lh, ls, lv], np.uint8), np.array([179, us, uv], np.uint8))
    m2 = cv2.inRange(hsv, np.array([0, ls, lv], np.uint8), np.array([uh, us, uv], np.uint8))
    return cv2.bitwise_or(m1, m2)

def blob_analysis(img, morph_kernel=(3,3), morph_iters=1, min_pixels=1, rel_area_multiplier=0.004, max_components=2):
    """
    Return annotated image, counts, and masks for detected red and blue patches.
    Keeps only up to `max_components` largest components per color.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV ranges (tune if needed)
    red_lower1 = np.array([0, 100, 50], dtype=np.uint8)
    red_upper1 = np.array([1, 255, 255], dtype=np.uint8) #10 original
    red_lower2 = np.array([170, 100, 50], dtype=np.uint8) #170 original
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    blue_lower = np.array([100, 100, 50], dtype=np.uint8)
    blue_upper = np.array([130, 255, 255], dtype=np.uint8)

    # Build masks
    mask_red = cv2.bitwise_or(color_mask(hsv, red_lower1, red_upper1),
                              color_mask(hsv, red_lower2, red_upper2))
    mask_blue = color_mask(hsv, blue_lower, blue_upper)

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

def crop_top_of_roi(roi):
    """Crop the top portion of a person ROI to focus on upper body where armband is."""
    TOP_REMOVE_RATIO = 0.25  # remove top 25% of ROI
    TOP_KEEP_RATIO = 0.5     # keep up to 50% of ROI
    h = roi.shape[0]
    start_row = int(h * TOP_REMOVE_RATIO)
    end_row = int(h * TOP_KEEP_RATIO)
    if start_row >= end_row:
        roi = roi[:int(h * TOP_KEEP_RATIO), :]
    else:
        roi = roi[start_row:end_row, :]

    SIDES_REMOVE_RATIO = 0.15  # remove 15% from left and 15% from right
    w = roi.shape[1]
    left = int(w * SIDES_REMOVE_RATIO)
    right = int(w * (1.0 - SIDES_REMOVE_RATIO))
    roi = roi[:, left:right]
    return roi

def edge_detection(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = cv2.magnitude(sobelx, sobely)

    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    return gradient_magnitude


def process_person_roi(roi, person_idx):
    """Process a single person's ROI and return classification."""

    # Crop to upper body area
    #cropped = crop_top_of_roi(roi)
    
    # Adaptive blur based on ROI size - larger ROIs need more blur
    h, w = roi.shape[:2] #cropped.shape[:2]
    roi_area = h * w
    
    # Scale kernel size based on area (min 5x5, increase for larger ROIs)
    if roi_area < 10000:  # Small ROI
        kernel_size = 5
    elif roi_area < 20000:  # Medium ROI
        kernel_size = 7
    elif roi_area < 300000:  # Large ROI
        kernel_size = 9
    else:  # Very large ROI
        kernel_size = 11
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

    if roi_area > 5000:
        edge = edge_detection(blurred)
        cv2.imshow("Edges", edge)

    annotated, mask_red, mask_blue, red_boxes, blue_boxes = blob_analysis(
        blurred, 
        morph_kernel=(3,3), 
        morph_iters=1,
        min_pixels=10,
        rel_area_multiplier=0.002,
        max_components=2
    )
    
    # Classify the target based on detected boxes
    classification, target_type, is_hvt = classify_target(red_boxes, blue_boxes)
    
    return annotated, classification, target_type, is_hvt, len(red_boxes), len(blue_boxes)

def process_frame_with_boxes(img, frame_number, annotations):
    """Process a frame with given bounding boxes, analyzing each person separately."""
    results = []
    display_images = []
    
    if not annotations:
        print(f"Frame {frame_number}: No bounding boxes found")
        return img, results, display_images
    
    print(f"\nFrame {frame_number}: Processing {len(annotations)} person(s)")
    
    output_img = img.copy()
    
    for idx, ann in enumerate(annotations, 1):
        bbox = ann['bbox']  # COCO format: [x, y, width, height]
        x, y, w, h = [int(v) for v in bbox]
        
        # Extract ROI
        roi = img[y:y+h, x:x+w]
        
        if roi.size == 0:
            print(f"  Person {idx}: Invalid ROI, skipping")
            continue
        
        # Process this person's ROI
        annotated, classification, target_type, is_hvt, n_red, n_blue = process_person_roi(roi, idx)
        
        # Print classification
        print(f"  Person {idx}: {classification} | Red: {n_red}, Blue: {n_blue}")
        
        # Draw bounding box on output image
        color = (0, 255, 0) if target_type == "good" else (0, 0, 255) if target_type == "bad" else (128, 128, 128)
        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
        
        # Add label
        label = f"P{idx}: {classification}"
        cv2.putText(output_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Store the annotated ROI for display
        display_images.append((f"Person {idx}", annotated))
        
        results.append({
            'person_id': idx,
            'classification': classification,
            'target_type': target_type,
            'is_hvt': is_hvt,
            'bbox': (x, y, w, h)
        })
    
    return output_img, results, display_images
def show_video(video_path, json_path=None):
    """Process and display video frame by frame with optional bounding boxes from JSON."""
    # Load annotations if JSON provided
    frame_annotations = {}
    image_info = {}
    if json_path and os.path.exists(json_path):
        print(f"Loading annotations from: {json_path}")
        frame_annotations, image_info = load_coco_annotations(json_path)
    else:
        print("No JSON file provided or file not found - will skip frames without annotations")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    win_main = "Video - Press SPACE for next frame, Q to quit"
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        try:
            # Try to match frame to image_id (same logic as overlay_coco_annotations.py)
            matching_annotations = []
            
            # Try different possible image_id values to find the match
            for possible_id in [frame_idx, frame_idx + 1, 3720 + frame_idx]:
                if possible_id in frame_annotations:
                    matching_annotations = frame_annotations[possible_id]
                    break
            
            if matching_annotations:
                # Process with bounding boxes
                out, results, display_images = process_frame_with_boxes(frame, frame_idx, matching_annotations)
                
                # Display main frame
                cv2.imshow(win_main, out)
                
                # Display individual person ROIs
                for person_label, roi_img in display_images:
                    win_name = f"{person_label} - Frame {frame_idx}"
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(win_name, roi_img)
            else:
                # No boxes for this frame, just show the frame
                if frame_idx % 30 == 0:  # Print less frequently
                    print(f"Frame {frame_idx}: No annotations")
                cv2.imshow(win_main, frame)
        
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Wait for keypress (0 = wait indefinitely)
        k = cv2.waitKey(0) & 0xFF
        if k == 27 or k == ord('q'):  # ESC or Q to quit
            break
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = r"ProjektVideoer/3 mili 2 onde 1 god.MP4"
    COCO_JSON = r"Testing/3mili 2 onde 1 god.json"
    
    video_path = get_video_path(video_file)
    show_video(video_path, COCO_JSON)