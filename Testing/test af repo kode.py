import cv2
import joblib
import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_recall_curve, auc

# ---------------- USER CONFIG ----------------
VIDEO_PATHS = [
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 mili en idiot der ligger ned.MP4",
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\3 mili 2 onde 1 god.MP4"
]

GROUND_TRUTH_JSONS = [
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json",
    r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\3mili 2 onde 1 god.json"
]

MODEL_PATH = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Person_Detector_Json+YOLO.pkl"
OUTPUT_DIR = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\test_plots_excel_1"

WINDOW_SIZE = (128, 256)
DETECTION_STEP = 16
CONF_THRESHOLD = -999999999999999
IOU_THRESHOLD = 0.5
USE_NMS = True
NMS_IOU = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- HELPER FUNCTIONS ----------------
def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = {img_id: [] for img_id in images.keys()}

    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann["bbox"])

    return images, anns_by_image


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter

    return inter / union if union > 0 else 0


def nms(detections, iou_thr):
    if len(detections) == 0:
        return []

    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best["bbox"], d["bbox"]) < iou_thr]

    return keep


def match_detections(dets, gts, iou_thr):
    matched_gt = set()
    TP_scores, FP_scores = [], []

    for det in dets:
        best_iou = 0
        best_gt_idx = None

        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue

            iou_val = iou(det["bbox"], gt)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = i

        if best_iou >= iou_thr:
            matched_gt.add(best_gt_idx)
            TP_scores.append(det["score"])
        else:
            FP_scores.append(det["score"])

    FN_count = len(gts) - len(matched_gt)
    return TP_scores, FP_scores, FN_count


# ---------------- MAIN PROGRAM ----------------
def main():
    # Load model
    model_data = joblib.load(MODEL_PATH)
    clf = model_data.get("classifier", model_data) if isinstance(model_data, dict) else model_data

    # Pre-create HOG descriptor
    hog = cv2.HOGDescriptor(
        _winSize=WINDOW_SIZE,
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(8, 8),
        _nbins=9
    )

    all_TP, all_FP = [], []
    total_FN = 0

    # Count total frames for ETA
    total_frames_all = 0
    for v in VIDEO_PATHS:
        cap = cv2.VideoCapture(v)
        total_frames_all += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    processed_frames = 0
    start_time_total = time.time()

    # Loop datasets
    for video_path, coco_path in zip(VIDEO_PATHS, GROUND_TRUTH_JSONS):

        images, anns_by_image = load_coco_annotations(coco_path)
        frame_map = {img["file_name"]: img_id for img_id, img in images.items()}

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            processed_frames += 1

            filename = f"{frame_idx}.jpg"
            if filename not in frame_map:
                continue

            img_id = frame_map[filename]
            gts = anns_by_image[img_id]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape

            detections = []

            # Sliding window
            for y in range(0, H - WINDOW_SIZE[1], DETECTION_STEP):
                for x in range(0, W - WINDOW_SIZE[0], DETECTION_STEP):
                    patch = gray[y:y + WINDOW_SIZE[1], x:x + WINDOW_SIZE[0]]
                    feat = hog.compute(patch).reshape(1, -1)
                    score = clf.decision_function(feat)[0]

                    if score >= CONF_THRESHOLD:
                        detections.append({"bbox": [x, y, WINDOW_SIZE[0], WINDOW_SIZE[1]], "score": score})

            if USE_NMS:
                detections = nms(detections, NMS_IOU)

            TP_scores, FP_scores, FN_count = match_detections(detections, gts, IOU_THRESHOLD)

            all_TP.extend(TP_scores)
            all_FP.extend(FP_scores)
            total_FN += FN_count

            # Progress
            if processed_frames % 100 == 0:
                elapsed = time.time() - start_time_total
                pct = processed_frames / total_frames_all
                eta = elapsed / pct - elapsed
                print(f"\n{processed_frames}/{total_frames_all} frames ({pct*100:.1f}%)")
                print(f"Elapsed: {int(elapsed)}s, ETA: {int(eta)}s")
                print(f"TP={len(all_TP)}, FP={len(all_FP)}, FN={total_FN}")

        cap.release()

    print("\nTesting complete.")

    # ---------------- METRICS ----------------
    y_scores = np.array(all_TP + all_FP)
    y_true = np.array([1] * len(all_TP) + [0] * len(all_FP))

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    print("DEBUG: TP:", len(all_TP), "FP:", len(all_FP), "FN:", total_FN)

    # Save plot
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUC={pr_auc:.3f})")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "PR_Curve.png"))

    # Save CSV
    with open(os.path.join(OUTPUT_DIR, "PR_data.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "threshold"])
        for p, r, t in zip(precision, recall, np.append(thresholds, np.nan)):
            writer.writerow([p, r, t])

    print(f"AUC: {pr_auc:.4f}")
    print(f"TP={len(all_TP)}, FP={len(all_FP)}, FN={total_FN}")


if __name__ == "__main__":
    main()
