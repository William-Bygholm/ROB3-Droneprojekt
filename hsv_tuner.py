import cv2
import numpy as np
import os
import glob
import json

# --- Config: folder / single image to load ---
IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), "mili_med_og_uden_bond")
DEFAULT_IMAGE = "mili_med_og_uden_bond/mili 24.png"  # set to full path to a single image to force one image

# --- Only red and blue ranges ---
DEFAULT_RANGES = {
    "red":  {"lower": [0, 100, 50],  "upper": [10, 255, 255], "on": 1},
    "blue": {"lower": [100, 100, 50], "upper": [130, 255, 255], "on": 1},
}

SLIDER_WINDOW = "HSV Sliders"
PREVIEW_WINDOW = "HSV Preview"
SAVE_FILE = os.path.join(os.path.dirname(__file__), "hsv_ranges.json")


def load_image():
    if DEFAULT_IMAGE and os.path.isfile(DEFAULT_IMAGE):
        return cv2.imread(DEFAULT_IMAGE)
    if not os.path.isdir(IMAGES_FOLDER):
        raise FileNotFoundError(IMAGES_FOLDER)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(IMAGES_FOLDER, e)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images in {IMAGES_FOLDER}")
    return cv2.imread(files[0])


def nothing(x):
    pass


def make_trackbars(ranges):
    """Create trackbars inside SLIDER_WINDOW (separate popup for sliders)."""
    cv2.namedWindow(SLIDER_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SLIDER_WINDOW, 360, 300)
    try:
        cv2.moveWindow(SLIDER_WINDOW, 20, 40)
    except Exception:
        pass
    for cname, vals in ranges.items():
        prefix = cname.upper()[:3]
        cv2.createTrackbar(f"{prefix}_ON", SLIDER_WINDOW, vals.get("on", 1), 1, nothing)
        lh, ls, lv = vals["lower"]
        cv2.createTrackbar(f"{prefix}_L_H", SLIDER_WINDOW, int(lh), 179, nothing)
        cv2.createTrackbar(f"{prefix}_L_S", SLIDER_WINDOW, int(ls), 255, nothing)
        cv2.createTrackbar(f"{prefix}_L_V", SLIDER_WINDOW, int(lv), 255, nothing)
        uh, us, uv = vals["upper"]
        cv2.createTrackbar(f"{prefix}_U_H", SLIDER_WINDOW, int(uh), 179, nothing)
        cv2.createTrackbar(f"{prefix}_U_S", SLIDER_WINDOW, int(us), 255, nothing)
        cv2.createTrackbar(f"{prefix}_U_V", SLIDER_WINDOW, int(uv), 255, nothing)


def read_ranges_from_trackbars(ranges):
    out = {}
    for cname in ranges.keys():
        prefix = cname.upper()[:3]
        on = cv2.getTrackbarPos(f"{prefix}_ON", SLIDER_WINDOW)
        lh = cv2.getTrackbarPos(f"{prefix}_L_H", SLIDER_WINDOW)
        ls = cv2.getTrackbarPos(f"{prefix}_L_S", SLIDER_WINDOW)
        lv = cv2.getTrackbarPos(f"{prefix}_L_V", SLIDER_WINDOW)
        uh = cv2.getTrackbarPos(f"{prefix}_U_H", SLIDER_WINDOW)
        us = cv2.getTrackbarPos(f"{prefix}_U_S", SLIDER_WINDOW)
        uv = cv2.getTrackbarPos(f"{prefix}_U_V", SLIDER_WINDOW)
        lower = np.array([lh, ls, lv], dtype=np.uint8)
        upper = np.array([uh, us, uv], dtype=np.uint8)
        out[cname] = {"lower": lower, "upper": upper, "on": bool(on)}
    return out


def visualize(img, combined_bg_mask, fg_mask, masked_img):
    # convert masks to BGR for stacking
    bg_bgr = cv2.cvtColor(combined_bg_mask, cv2.COLOR_GRAY2BGR)
    fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    # annotate
    h = 25
    def label(img_in, txt):
        out = img_in.copy()
        cv2.rectangle(out, (0,0), (out.shape[1], h), (0,0,0), -1)
        cv2.putText(out, txt, (6, h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return out
    orig = label(img, "Original")
    bg_bgr = label(bg_bgr, "Combined Mask (red + blue)")
    fg_bgr = label(fg_bgr, "Foreground mask (inverted)")
    masked_label = label(masked_img, "Masked image (background removed)")
    top = np.hstack([orig, bg_bgr])
    bot = np.hstack([fg_bgr, masked_label])
    # resize if too large to fit screen
    maxw = 1400
    scale = 1.0
    if top.shape[1] > maxw:
        scale = maxw / top.shape[1]
    if scale != 1.0:
        top = cv2.resize(top, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bot = cv2.resize(bot, (top.shape[1], int(bot.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return np.vstack([top, bot])


def save_ranges(ranges):
    serializable = {}
    for k,v in ranges.items():
        serializable[k] = {
            "lower": [int(x) for x in v["lower"]],
            "upper": [int(x) for x in v["upper"]],
            "on": int(v["on"])
        }
    with open(SAVE_FILE, "w") as f:
        json.dump(serializable, f, indent=2)
    print("Saved ranges to", SAVE_FILE)


def main():
    img = load_image()
    # create separate slider window
    make_trackbars(DEFAULT_RANGES)

    # create preview window separate from sliders
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, 1000, 700)
    try:
        cv2.moveWindow(PREVIEW_WINDOW, 420, 40)
    except Exception:
        pass

    while True:
        ranges = read_ranges_from_trackbars(DEFAULT_RANGES)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # build combined background mask from selected colors
        combined_bg = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for cname, info in ranges.items():
            if not info["on"]:
                continue
            lower = info["lower"]
            upper = info["upper"]
            mask = cv2.inRange(hsv, lower, upper)
            combined_bg = cv2.bitwise_or(combined_bg, mask)

        # optional morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        combined_bg = cv2.morphologyEx(combined_bg, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined_bg = cv2.morphologyEx(combined_bg, cv2.MORPH_OPEN, kernel, iterations=1)

        fg_mask = cv2.bitwise_not(combined_bg)
        masked = cv2.bitwise_and(img, img, mask=fg_mask)

        vis = visualize(img, combined_bg, fg_mask, masked)
        cv2.imshow(PREVIEW_WINDOW, vis)

        key = cv2.waitKey(50) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord("s"):
            # save current numeric ranges
            out_ranges = {}
            for k,v in ranges.items():
                out_ranges[k] = {
                    "lower": v["lower"].tolist(),
                    "upper": v["upper"].tolist(),
                    "on": int(v["on"])
                }
            with open(SAVE_FILE, "w") as f:
                json.dump(out_ranges, f, indent=2)
            print("Saved ranges to", SAVE_FILE)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()