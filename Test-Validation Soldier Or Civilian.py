import cv2
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix

VIDEO_PATH = "ProjektVideoer/2 militær med blå bånd .MP4"
COCO_JSON = "Validation/2 mili med blå bond.json"

def compute_histogram(img, center_y_ratio=0.35, center_x_ratio=0.5, height_ratio=0.2, width_ratio=0.3):
    """
    Help-function to compute a normalized HSV histogram for the upper part (breast region) of an image.
    This is used both for reference histograms creation and for classification.
    """
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

def classify_person(roi, reference_histograms, method=cv2.HISTCMP_BHATTACHARYYA, threshold_score=0.8):
    """
    Classify a person in the ROI as 'soldier' or 'unkown' based on histogram comparison.
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
        print(f"Classification: {best_label}")
        return best_label
    else:
        print(f"No military match found. Best score: {best_score}")
        print(f"Classification: Civilian")
        return "Civilian"


# Validation and test

def show_video_with_annotations(video_path=VIDEO_PATH, json_path=COCO_JSON, id_offsets=(0, 1),
                                center_y_ratio=0.35, center_x_ratio=0.5,
                                height_ratio=0.2, width_ratio=0.3):
    """
    Viser video med bounding boxes fra JSON og crop overlay (det område der bruges til histogram).
    Ingen resize — crop beregnes direkte på ROI'ens størrelse.
    Tryk 'q' for at stoppe afspilningen.
    """
    CLASS_COLORS = {
        "Military good": (255, 0, 0),
        "Military bad": (0, 0, 255),
        "Good HVT": (255, 255, 0),
        "Bad HVT": (0, 128, 255),
        "Civilian": (128, 128, 128),
        "Unknown person": (0, 255, 255)
    }

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frame_annotations = {}
    for ann in data.get("annotations", []):
        iid = ann.get("image_id")
        if iid is None:
            continue
        frame_annotations.setdefault(iid, []).append(ann)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        anns = []
        for off in id_offsets:
            if (frame_idx + off) in frame_annotations:
                anns = frame_annotations[frame_idx + off]
                break

        for ann in anns:
            x, y, w, h = [int(v) for v in ann["bbox"]]

            # Klasse og farve
            attrs = ann.get("attributes", {})
            class_name = "Unknown person"
            for cname, active in attrs.items():
                if active and cname in CLASS_COLORS:
                    class_name = cname
                    break
            color = CLASS_COLORS.get(class_name, (255,255,255))

            # Tegn ROI-boks
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, class_name, (x, max(0,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Crop overlay direkte i ROI
            roi_h, roi_w = h, w
            if roi_h > 0 and roi_w > 0:
                crop_h = max(1, int(roi_h * height_ratio))
                crop_w = max(1, int(roi_w * width_ratio))
                y_center = int(roi_h * center_y_ratio)
                x_center = int(roi_w * center_x_ratio)

                y_start = max(0, y_center - crop_h // 2)
                y_end   = min(roi_h, y_start + crop_h)
                x_start = max(0, x_center - crop_w // 2)
                x_end   = min(roi_w, x_start + crop_w)

                # Koordinater i hele frame
                cx1 = x + x_start
                cy1 = y + y_start
                cx2 = x + x_end
                cy2 = y + y_end

                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0,255,0), 2)

        # Overlay info
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        display_frame = cv2.resize(frame, (0,0), fx=0.75, fy=0.75)
        cv2.imshow("Annotated Video + Crop Overlay", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

# Main
reference_histograms = load_reference_histograms("Reference templates")
show_video_with_annotations()