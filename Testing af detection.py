import json
import cv2
import time
from SVM_In_Action import detect_people, clf, hog
from tqdm import tqdm  # progress bar

IOU_THRESHOLD = 0.5
FRAME_WIDTH = 640   # resize frame for faster detection
FRAME_HEIGHT = 360

VIDEOS = [
    (r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 mili en idiot der ligger ned.MP4",
     r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json"),
    (r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\3 mili 2 onde 1 god.MP4",
     r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\3mili 2 onde 1 god.json"),
]

# ---------------- IoU ----------------
def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# ---------------- Load CVAT JSON ----------------
def load_cvat_annotations(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: could not read/parse JSON '{json_file}': {e}")
        return {}

    ann_by_frame = {}

    if not data:
        print(f"Warning: annotation file '{json_file}' is empty.")
        return {}

    # COCO format support
    if isinstance(data, dict) and "images" in data and "annotations" in data:
        # Map image_id to frame index (assume sorted by id)
        image_id_to_frame = {}
        for idx, img in enumerate(sorted(data["images"], key=lambda x: x["id"])):
            image_id_to_frame[img["id"]] = idx

        for ann in data["annotations"]:
            image_id = ann.get("image_id")
            bbox = ann.get("bbox")  # [x, y, width, height]
            if image_id is None or bbox is None or len(bbox) != 4:
                continue
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            frame_idx = image_id_to_frame.get(image_id)
            if frame_idx is None:
                continue
            box = [x1, y1, x2, y2]
            ann_by_frame.setdefault(frame_idx, []).append(box)
        return ann_by_frame

    # ...existing code for CVAT and other formats...
    # Case 1: CVAT-like 'shapes' list
    if isinstance(data, dict) and "shapes" in data and isinstance(data["shapes"], list):
        for shape in data["shapes"]:
            frame_idx = shape.get("frame")
            pts = shape.get("points")
            if frame_idx is None or not pts or len(pts) < 2:
                continue
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            box = [x1, y1, x2, y2]
            ann_by_frame.setdefault(frame_idx, []).append(box)
        return ann_by_frame

    if isinstance(data, list):
        for shape in data:
            frame_idx = shape.get("frame")
            pts = shape.get("points")
            if frame_idx is None or not pts or len(pts) < 2:
                continue
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            box = [x1, y1, x2, y2]
            ann_by_frame.setdefault(frame_idx, []).append(box)
        return ann_by_frame

    if isinstance(data, dict) and "annotations" in data and isinstance(data["annotations"], list):
        for shape in data["annotations"]:
            frame_idx = shape.get("frame")
            pts = shape.get("points") or shape.get("bbox")
            if frame_idx is None or not pts:
                continue
            if isinstance(pts[0], list) and len(pts) >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
            else:
                x1, y1, x2, y2 = pts[:4]
            box = [x1, y1, x2, y2]
            ann_by_frame.setdefault(frame_idx, []).append(box)
        return ann_by_frame

    print(f"Warning: unknown annotation format in '{json_file}'. Expected COCO, CVAT, or a list of annotations.")
    return {}

# ---------------- Evaluation ----------------
TP = 0
FP = 0
FN = 0

for video_path, json_path in VIDEOS:
    print(f"\n=== Evaluating: {video_path} ===")
    gt = load_cvat_annotations(json_path)
    annotated_frames = set(gt.keys())
    print(f"Frames with ground truth annotations: {len(annotated_frames)}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()


    for frame_idx in tqdm(range(total_frames), desc="Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        # --- resize for faster detection ---
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        if frame_idx in annotated_frames:
            pred_boxes = detect_people(frame, clf, hog)
            gt_boxes = gt.get(frame_idx, [])

            matched_gt = set()
            for pb in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for idx, gb in enumerate(gt_boxes):
                    val = iou(pb, gb)
                    if val > best_iou:
                        best_iou = val
                        best_gt_idx = idx

                if best_iou >= IOU_THRESHOLD:
                    TP += 1
                    matched_gt.add(best_gt_idx)
                else:
                    FP += 1  # FP i annoteret frame

            FN += len(gt_boxes) - len(matched_gt)
        else:
            # --- FP på frames uden annotationer ---
            pred_boxes = detect_people(frame, clf, hog)
            FP += len(pred_boxes)

        # ETA
        elapsed = time.time() - start_time
        pct_done = (frame_idx + 1) / total_frames * 100
        eta = elapsed / (frame_idx + 1) * (total_frames - frame_idx - 1)
        tqdm.write(f"Progress: {pct_done:.2f}% | ETA: {eta:.1f}s | TP: {TP} | FP: {FP} | FN: {FN} | Frame: {frame_idx+1}/{total_frames}", end="\r")

    cap.release()
    print(f"\nVideo finished. TP: {TP}, FP: {FP}, FN: {FN}")

# ---------------- Results ----------------
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)

print("\n======= FINAL RESULTS =======")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("=============================")
