import cv2
import json
import numpy as np
import csv
import os
from sklearn.metrics import precision_recall_curve, auc

# Importér din rigtige detector
from SVM_In_Action import detect_people, clf, hog

# ---------------- USER CONFIG ----------------
VIDEO_PATHS = [
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 mili en idiot der ligger ned.MP4",
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\3 mili 2 onde 1 god.MP4"
]

GROUND_TRUTH_JSONS = [
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json",
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\3mili 2 onde 1 god.json"
]

OUTPUT_DIR = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\test_plots_excel_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IOU_THRESHOLD = 0.5


# ---------------- HELPER FUNCTIONS ----------------
def load_coco(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns = {}
    for ann in data["annotations"]:
        anns.setdefault(ann["image_id"], []).append(ann["bbox"])

    return images, anns


def iou_xyxy(a, b):
    # a,b = [x1,y1,x2,y2]
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


def coco_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def match_detections(det_boxes, gt_boxes):
    matched = set()
    TP, FP = [], []

    for det in det_boxes:
        best_iou = 0
        best_idx = None

        for i, gt in enumerate(gt_boxes):
            if i in matched:
                continue
            iou_val = iou_xyxy(det, gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i

        if best_iou >= IOU_THRESHOLD:
            matched.add(best_idx)
            TP.append(1)
        else:
            FP.append(1)

    FN = len(gt_boxes) - len(matched)
    return TP, FP, FN


# ---------------- MAIN ----------------
def main():
    all_scores = []
    all_labels = []
    total_FN = 0

    for video_path, json_path in zip(VIDEO_PATHS, GROUND_TRUTH_JSONS):

        images, anns = load_coco(json_path)
        frame_map = {img["file_name"]: img_id for img_id, img in images.items()}

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            fname = f"{frame_idx}.jpg"

            if fname not in frame_map:
                continue

            img_id = frame_map[fname]
            gt_xyxy = [coco_to_xyxy(b) for b in anns.get(img_id, [])]

            # ✅ Brug din rigtige detector
            det_xyxy = detect_people(frame, clf, hog)

            # ✅ Match TP/FP/FN
            TP, FP, FN = match_detections(det_xyxy, gt_xyxy)
            total_FN += FN

            # ✅ Scores = antal bokse (vi bruger bare 1.0 for TP og 0.5 for FP)
            # Hvis du vil bruge SVM-scores, skal detect_people returnere dem.
            all_scores.extend([1.0] * len(TP))
            all_scores.extend([0.5] * len(FP))

            all_labels.extend([1] * len(TP))
            all_labels.extend([0] * len(FP))

        cap.release()

    # ---------------- PR CURVE ----------------
    if len(all_labels) == 0:
        print("ERROR: No detections at all. Something is wrong.")
        return

    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    pr_auc = auc(recall, precision)

    # Save CSV
    with open(os.path.join(OUTPUT_DIR, "PR_data.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["precision", "recall", "threshold"])
        for p, r, t in zip(precision, recall, np.append(thresholds, np.nan)):
            w.writerow([p, r, t])

    print(f"\n✅ DONE")
    print(f"AUC = {pr_auc:.4f}")
    print(f"FN total = {total_FN}")


if __name__ == "__main__":
    main()
