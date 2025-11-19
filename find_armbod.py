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


def remove_background_and_count(img, morph_kernel=(5,5), morph_iters=1, min_pixels=100, rel_area_multiplier=0.0005):
    """
    Return annotated image, counts, and masks for detected red and blue patches.
    Keeps both red and blue masks (does NOT invert).

    Parameters:
    - min_pixels: absolute minimum area (in pixels) for a component to be considered a band.
    - rel_area_multiplier: fraction of image area used to compute a relative minimum area.
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
    h, w = img.shape[:2]
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

    # Annotate original image with detections and counts
    out = img.copy()
    for (x, y, ww, hh, area) in red_boxes:
        cv2.rectangle(out, (x, y), (x+ww, y+hh), (0,0,255), 2)
        cv2.putText(out, f"R:{area}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    for (x, y, ww, hh, area) in blue_boxes:
        cv2.rectangle(out, (x, y), (x+ww, y+hh), (255,0,0), 2)
        cv2.putText(out, f"B:{area}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    summary = f"Red: {len(red_boxes)}  Blue: {len(blue_boxes)}"
    cv2.putText(out, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    return out, mask_red, mask_blue, red_boxes, blue_boxes


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # crop top half early to reduce false positives
    CROP_RATIO = 0.5
    h = img.shape[0]
    img = img[:int(h * CROP_RATIO), :]

    # blur color image (dynamic_blur handles multi-channel)
    blurred = dynamic_blur(img, scale=0.02, min_k=3, max_k=51)

    annotated, mask_red, mask_blue, red_boxes, blue_boxes = remove_background_and_count(blurred,
                                                                                      morph_kernel=(5,5),
                                                                                      morph_iters=1)
    # return annotated image (you may also return masks if wanted)
    return annotated


def dynamic_blur(gray, scale=0.1, min_k=1, max_k=101):
    """
    Compute a Gaussian blur kernel proportional to image size.
    - scale: fraction of the smaller image dimension used for kernel (e.g. 0.02 = 2%)
    - min_k/max_k: bounds for kernel size (must be odd). Returns blurred image.
    """
    h, w = gray.shape[:2]
    k = max(min_k, int(min(h, w) * scale))
    if k % 2 == 0:
        k += 1
    # ensure max_k odd
    if max_k % 2 == 0:
        max_k -= 1
    k = min(k, max_k)
    # ensure sensible minimum
    if k < 3:
        k = 3
    return cv2.GaussianBlur(gray, (k, k), 0)

def resize_and_pad(img, size=(800,600)):
    tgt_w, tgt_h = size
    h, w = img.shape[:2]
    scale = min(tgt_w / w, tgt_h / h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        canvas = np.zeros((tgt_h, tgt_w), dtype=resized.dtype)
    else:
        canvas = np.zeros((tgt_h, tgt_w, 3), dtype=resized.dtype)
    x = (tgt_w - nw)//2
    y = (tgt_h - nh)//2
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
        disp = resize_and_pad(disp, target_size:=display_size)
        cv2.imshow(win, disp)
        k = cv2.waitKey(0) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cv2.destroyAllWindows()

# usage
if __name__ == "__main__":
    folder = "C:\\Users\\olafa\\Documents\\GitHub\\ROB3-Droneprojekt\\mili_med_og_uden_bond"  # specify your folder here
    image_paths = get_image_paths(folder)
    show_images(image_paths, display_size=(800,600))