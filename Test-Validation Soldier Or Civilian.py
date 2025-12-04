import cv2
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from matplotlib import pyplot as plt

VIDEO_PATH = "ProjektVideoer/2 militær med blå bånd .MP4"
COCO_JSON = "Validation/2 mili med blå bond.json"
THRESHOLD_SCORE = 0.8

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
    hist = cv2.normalize(hist, hist).astype("float32")
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
    best_score = float('inf')

    for histograms in reference_histograms.values():
        for ref_hist in histograms:
            score = cv2.compareHist(roi_hist, ref_hist, method)
            if score < best_score:
                best_score = score
    
    # Reverse logic for Bhattacharyya distance: lower=better to higher=better and normalize to [0, 1]:
    match_score = max(0.0, min(1.0, 1.0 - best_score))

    if best_score < threshold_score:
        print(f"Best score {best_score}")
        print(f"Classification: Military")
        return "Military", match_score
    else:
        print(f"No military match found. Best score: {best_score}")
        print(f"Classification: Civilian")
        return "Civilian", match_score

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

def load_annotations(json_path=COCO_JSON):
    """
    Load JSON and return dict: image_id -> list of annotations for that frame.
    Each annotation is expected to contain:
    - "bbox": [x, y, w, h]
    - "attributes": dict with class labels (e.g. "Military good": true/false).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frame_annotations = {}
    for ann in data.get("annotations", []):
        iid = ann.get("image_id")
        if iid is None:
            continue
        frame_annotations.setdefault(iid, []).append(ann)
    return frame_annotations

def get_ground_truth_label(ann):
    """Returns 1 for Military related persons, 0 for Civilian and unknown."""
    attrs = ann.get("attributes", {})
    for cname, active in attrs.items():
        if not active:
            continue
        if "Military" in cname or "HVT" in cname:
            return 1
        if "Civilian" in cname:
            return 0
        if "Unknown" in cname:
            return 0
    # If nothing matched, treat as civilian.
    return 0 # fallback

def collect_scores(video_path, frame_annotations, reference_histograms, id_offsets=(0,1)):
    """
    Run a video frame by frame and classify persons in ROIs.
    Returns:
        ground_truth_labels -> 0 = Civilian, 1 = Military/HVT
        match_scores        -> continuous scores for precision/recall curve
        predicted_labels    -> binary predictions (0/1)
    """
    cap = cv2.VideoCapture(video_path)
    ground_truth_labels, match_scores, predicted_labels = [], [], []
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
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            gt_label = get_ground_truth_label(ann)
            ground_truth_labels.append(gt_label)

            pred_label, match_score = classify_person(roi, reference_histograms)
            match_scores.append(match_score)

            if "Military" in pred_label or "HVT" in pred_label:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)

        frame_idx += 1

    cap.release()
    return ground_truth_labels, match_scores, predicted_labels

def plot_precision_recall(ground_truth_labels, match_scores):
    """
    Plot Precision-Recall curve based on collected scores and ground truth labels.
    """
    precision, recall, thresholds = precision_recall_curve(ground_truth_labels, match_scores)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve: Military vs Civilian")
    plt.show()

def evaluate_classify_person(video_path, json_path, reference_path="Reference templates"):
    """
    Evaluates the classication algorithm on a video with annotations.
    Output:
    - Confusion matrix
    - Classification report (precision, recall, f1-score)
    - Precision-Recall curve
    """
    # Load reference histograms and annotations
    reference_histograms = load_reference_histograms(reference_path)
    frame_annotations = load_annotations(json_path)

    # Collect scores
    ground_truth_labels, match_scores, predicted_labels = collect_scores(video_path, frame_annotations, reference_histograms)

    # Confusion matrix
    print("\nConfusion Matrix:\n", confusion_matrix(ground_truth_labels, predicted_labels))

    # Classification report
    print("\nClassification Report:\n", classification_report(ground_truth_labels, predicted_labels))

    # Precision-Recall curve
    plot_precision_recall(ground_truth_labels, match_scores)

def evaluate_multiple_videos_combined(video_json_pairs, reference_path="Reference templates"):
    """
    Evaluates  the classifier algorithm on multiple videos combined.
    Returns one combined confusion matrix, one classification report, and one precision-recall plot.
    """
    # Load reference histograms
    reference_histograms = load_reference_histograms(reference_path)

    all_ground_truth, all_match_scores, all_predicted = [], [], []

    for idx, (video_path, json_path) in enumerate(video_json_pairs, start=1):
        print(f"\n--- Processing video {idx}: {video_path} ---")
        frame_annotations = load_annotations(json_path)

        ground_truth_labels, match_scores, predicted_labels = collect_scores(
            video_path, frame_annotations, reference_histograms)

        all_ground_truth.extend(ground_truth_labels)
        all_match_scores.extend(match_scores)
        all_predicted.extend(predicted_labels)

    # Combined confusion matrix
    print("\nConfusion Matrix (combined):\n", confusion_matrix(all_ground_truth, all_predicted))

    # Combined classification report
    print("\nClassification Report (combined):\n", classification_report(all_ground_truth, all_predicted))
    # Reuse existing plot function
    plot_precision_recall(all_ground_truth, all_match_scores)

# Main
video_json_pairs = [
    ("ProjektVideoer/2 mili en idiot der ligger ned.MP4", "Testing/2 mili og 1 idiot.json"),
    ("ProjektVideoer/3 mili 2 onde 1 god.MP4", "Testing/3mili 2 onde 1 god.json")
]
evaluate_multiple_videos_combined(video_json_pairs)