# imports
# detector.py
import cv2
import joblib
import numpy as np 
import os


# HOG + SVM her

 
VIDEO_IN = r"ProjektVideoer/2 militær med blå bånd .MP4"
MODEL_FILE = "svm_hog_model.pkl_v3"
WINDOW_SIZE = (128, 256)
SCALES = [1.0, 0.8, 0.64]
STEP_SIZES = {1.0: 32, 0.8: 28, 0.64: 20}
NMS_THRESHOLD = 0.15
DISPLAY_SCALE = 0.3
FRAME_SKIP = 10
SVM_THRESHOLD = 1

# ---------------- LOAD MODEL ----------------
clf = joblib.load(MODEL_FILE) 

hog = cv2.HOGDescriptor(
    _winSize=WINDOW_SIZE,
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(8, 8),
    _nbins=9
)

# ---------------- HELPERS ----------------
def sliding_windows(img, step, win_size):
    w, h = win_size
    for y in range(0, img.shape[0] - h + 1, step):
        for x in range(0, img.shape[1] - w + 1, step):
            yield x, y, img[y:y+h, x:x+w]

def nms_opencv(detections, scores, score_threshold, nms_threshold):
    if len(detections) == 0:
        return []

    # Convert [x1, y1, x2, y2] → [x, y, w, h]
    boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in detections]
    scores = [float(s) for s in scores]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold, nms_threshold)
    if len(indices) == 0:
        return []

    # Flatten and return only the selected boxes
    indices = indices.flatten()
    return [detections[i] for i in indices]

def merge_close_boxes(boxes, iou_threshold=0.2):
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        group = [boxes[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            xx1, yy1, xx2, yy2 = boxes[j]

            # Compute IoU
            inter_x1 = max(x1, xx1)
            inter_y1 = max(y1, yy1)
            inter_x2 = min(x2, xx2)
            inter_y2 = min(y2, yy2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xx2 - xx1) * (yy2 - yy1)
            iou = inter_area / float(area1 + area2 - inter_area)

            if iou > iou_threshold:
                group.append(boxes[j])
                used[j] = True

        # Merge group into one box
        gx1 = min(b[0] for b in group)
        gy1 = min(b[1] for b in group)
        gx2 = max(b[2] for b in group)
        gy2 = max(b[3] for b in group)
        merged.append([gx1, gy1, gx2, gy2])

    return merged


def detect_people(frame, clf, hog):
    """
    Detect people in a frame using HOG+SVM.
    Returns: list of bounding boxes [[x1, y1, x2, y2], ...]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape[:2]

    detections = []
    scores = []

    for scale in SCALES:
        resized = cv2.resize(gray, None, fx=scale, fy=scale)
        step = STEP_SIZES[scale]

        scale_x = w0 / resized.shape[1]
        scale_y = h0 / resized.shape[0]

        for x, y, win in sliding_windows(resized, step, WINDOW_SIZE):
            if win.shape != (WINDOW_SIZE[1], WINDOW_SIZE[0]):
                continue
            feat = hog.compute(win).ravel()
            score = clf.decision_function([feat])[0]

            if score > SVM_THRESHOLD:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + WINDOW_SIZE[0]) * scale_x)
                y2 = int((y + WINDOW_SIZE[1]) * scale_y)
                detections.append([x1, y1, x2, y2])
                scores.append(score)

    nms_boxes = nms_opencv(detections, scores, SVM_THRESHOLD, NMS_THRESHOLD)
    final_boxes = merge_close_boxes(nms_boxes, iou_threshold=0.2)
    
    return final_boxes


# from Soldier Or Civilian her


def compute_histogram(img, target_size=(64, 128), center_y_ratio=0.4, center_x_ratio=0.5, height_ratio=0.3, width_ratio=0.3):
    """
    Help-function to compute a normalized HSV histogram for the upper part (breast region) of an image.
    This is used both for reference histograms creation and for classification.
    """
    img = cv2.resize(img, target_size)
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
    
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist)
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

def show_crop_overlay(img, target_size=(64, 128), center_y_ratio=0.4, center_x_ratio=0.5, height_ratio=0.3, width_ratio=0.3):
    """
    A function only to test and visualize the cropping area used in compute_histogram.
    """
    img = cv2.resize(img, target_size)
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

def classify_person(roi, reference_histograms, method=cv2.HISTCMP_BHATTACHARYYA, threshold_score=0.3):
    """
    Classify a person in the ROI as 'soldier' or 'unkown' based on histogram comparison.
    Returns: (classification, best_score)
    """
    roi_hist = compute_histogram(roi)
    best_label = None
    best_score = float('inf')

    for label, histograms in reference_histograms.items():
        for ref_hist in histograms:
            score = cv2.compareHist(roi_hist, ref_hist, method)
            if score < best_score:
                best_score = score
                best_label = label

    if best_score < threshold_score:
        print(f"Best score {best_score}")
        return "soldier"
    else:
        print(f"No military match found. Best score: {best_score}")
        return "unknown"

# find armbånd her

def get_image_paths(folder_path):
    folder_path = os.path.join(os.path.dirname(__file__), folder_path)
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path '{folder_path}' is not a valid directory.")
    files = sorted(os.listdir(folder_path))
    image_paths = [os.path.join(folder_path, f) for f in files
                   if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]]
    return image_paths

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
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170, 100, 50], dtype=np.uint8)
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
        return "No target", None, False
    
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
        # Compare total areas
        red_area = sum(box[4] for box in red_boxes)
        blue_area = sum(box[4] for box in blue_boxes)
        
        if blue_area > red_area:
            return "Good soldier (area-based, 2v2)", "good", True
        else:
            return "Bad soldier (area-based, 2v2)", "bad", True
    
    # Fallback for any other combinations
    return "Uncertain classification", None, False

def crop_image(img):
    # --- vertical crop: remove top portion and keep a top block ---
    TOP_REMOVE_RATIO = 0.25  # remove top 25% of original
    TOP_KEEP_RATIO = 0.5     # keep up to 50% of original (rows start_row .. end_row)
    h = img.shape[0]
    start_row = int(h * TOP_REMOVE_RATIO)
    end_row = int(h * TOP_KEEP_RATIO)
    if start_row >= end_row:
        img = img[:int(h * TOP_KEEP_RATIO), :]
    else:
        img = img[start_row:end_row, :]

    # --- horizontal crop: remove 5% from each side (margin) ---
    WIDTH_REMOVE_RATIO = 0.05  # remove 5% from left and 5% from right
    w = img.shape[1]
    left = int(w * WIDTH_REMOVE_RATIO)
    right = int(w * (1.0 - WIDTH_REMOVE_RATIO))
    if left < right:
        img = img[:, left:right]
    return img


# ---------------- VIDEO PROCESSING ----------------
OUTPUT_DIR = "Output/Detections"  # Specify output folder here
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create folder if it doesn't exist

cap = cv2.VideoCapture(VIDEO_IN)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    # Detect people and get bounding boxes
    final_boxes = detect_people(frame, clf, hog)

    # Draw boxes on frame
    orig_frame = frame.copy()
    
    # Load reference histograms once (move outside loop for efficiency in real code)
    reference_histograms = load_reference_histograms("Reference templates")
    
    # Process each detected person
    for idx, (x1, y1, x2, y2) in enumerate(final_boxes):
        # Extract the person ROI
        roi = frame[y1:y2, x1:x2]

        # Skip if ROI is too small
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        
        # Classify person as soldier or unknown
        classification = classify_person(roi, reference_histograms, threshold_score=0.8)
        
        if classification == "soldier":
            # Analyze for armbands
            cropped_roi = crop_image(roi)
            blurred = cv2.GaussianBlur(cropped_roi, (5,5), 0)
            cv2.imshow("olaf", blurred)
            annotated, mask_red, mask_blue, red_boxes, blue_boxes = blob_analysis(blurred, morph_kernel=(3,3), morph_iters=1)
            
            # Classify target based on armband colors
            target_class, target_type, is_hvt = classify_target(red_boxes, blue_boxes)
            
            # Draw box with appropriate color
            if target_type == "good":
                box_color = (255, 0, 0)  # Blue for good
            elif target_type == "bad":
                box_color = (0, 0, 255)  # Red for bad
            else:
                box_color = (0, 255, 255)  # Yellow for unknown
            
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), box_color, 20)
            
            # Add label
            label = f"{target_class}"
            cv2.putText(orig_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 20)
        
        elif classification == "unknown":
            # Draw gray box for civilians
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (128, 128, 128), 20)
            cv2.putText(orig_frame, "Civilian", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

    display_frame = cv2.resize(orig_frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Detection & Classification", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()