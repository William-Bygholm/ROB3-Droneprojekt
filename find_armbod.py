import os
import cv2
import numpy as np


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


def remove_background_and_count(img, morph_kernel=(3,3), morph_iters=1, min_pixels=1, rel_area_multiplier=0.004, max_components=2):
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
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    #mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=morph_iters)
    #mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    #mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=morph_iters)
    #mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)

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

    # Annotate the boosted image
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


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    img = crop_image(img)

    # blur color image 
    blurred = cv2.medianBlur(img, 3)
    blurred2 = cv2.GaussianBlur(blurred,(3,3),0)

    annotated, mask_red, mask_blue, red_boxes, blue_boxes = remove_background_and_count(blurred2, morph_kernel=(3,3), morph_iters=1)
    
    # Classify the target based on detected boxes
    classification, target_type, is_hvt = classify_target(red_boxes, blue_boxes)
    
    # Print classification to console
    print(f"Image: {os.path.basename(image_path)}")
    print(f"  Classification: {classification}")
    print(f"  Red boxes: {len(red_boxes)} | Blue boxes: {len(blue_boxes)}")
    print()
    
    # return annotated image (you may also return masks if wanted)
    return annotated

def resize_and_pad(img, target_size=(800,600)):
    """
    Resize img to fit inside target_size while keeping aspect ratio.
    Pads with black to exactly match target_size.
    """
    tgt_w, tgt_h = target_size
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((tgt_h, tgt_w), dtype=img.dtype) if img.ndim == 2 else np.zeros((tgt_h, tgt_w, 3), dtype=img.dtype)
    scale = min(tgt_w / w, tgt_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        canvas = np.zeros((tgt_h, tgt_w), dtype=resized.dtype)
    else:
        canvas = np.zeros((tgt_h, tgt_w, 3), dtype=resized.dtype)
    x = (tgt_w - nw) // 2
    y = (tgt_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def show_images(paths, display_size=(800,600)):
    if not paths:
        print("No images found")
        return
    win = "Img"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for p in paths:
        try:
            out = process_image(p)
        except Exception as e:
            print("skip:", p, e)
            continue
        # ensure 3-channel for consistent display
        if out.ndim == 2:
            disp = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        else:
            disp = out
        disp = resize_and_pad(disp, target_size=display_size)
        cv2.imshow(win, disp)
        k = cv2.waitKey(0) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder = "mili_med_og_uden_bond"
    image_paths = get_image_paths(folder)
    show_images(image_paths, display_size=(800,600))