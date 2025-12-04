import cv2
import joblib
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# ---------------- USER CONFIG ----------------
DATASETS = [
    {
        "video": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\2 mili en idiot der ligger ned.MP4",
        "coco_json": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\2 mili og 1 idiot.json"
    },
    {
        "video": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\ProjektVideoer\3 mili 2 onde 1 god.MP4",
        "coco_json": r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Testing\3mili 2 onde 1 god.json"
    }
]

MODEL_FILE = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Person_Detector_Json+YOLO.pkl"
OUTPUT_DIR = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\test_plots_excel_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "combined_test_results.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "Combined_PR.png")

# HOG / SVM parameters
WIN_SIZE = (128, 256)
BLOCK_SIZE = (32, 32)
BLOCK_STRIDE = (16, 16)
CELL_SIZE = (8, 8)
NBINS = 9
STEP_SIZE = 48
CONF_THRESHOLD = 0.0  # keep everything, filter later
IOU_POSITIVE = 0.5

DEBUG = True

# ---------------- HELPER FUNCTIONS ----------------
def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = {img_id: [] for img_id in images.keys()}
    for ann in coco.get("annotations", []):
        if ann["category_id"] != 1:
            continue
        anns_by_image[ann["image_id"]].append(ann["bbox"])
    return images, anns_by_image

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2])
    y2 = min(a[1]+a[3], b[1]+b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0

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

# ---------------- MAIN ----------------
def main():
    model_data = joblib.load(MODEL_FILE)
    clf = model_data.get("classifier", model_data) if isinstance(model_data, dict) else model_data

    all_TP, all_FP = [], []
    total_FN = 0

    hog = cv2.HOGDescriptor(_winSize=WIN_SIZE, _blockSize=BLOCK_SIZE,
                            _blockStride=BLOCK_STRIDE, _cellSize=CELL_SIZE,
                            _nbins=NBINS)

    for dataset in DATASETS:
        video_path = dataset["video"]
        coco_path  = dataset["coco_json"]

        if DEBUG:
            print("\n--- Loading dataset ---")
            print("Video:", video_path)
            print("JSON :", coco_path)

        images, anns_by_image = load_coco_annotations(coco_path)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx + 1) not in images:
                frame_idx += 1
                continue

            gts = anns_by_image[frame_idx + 1]

            detections = []
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for y in range(0, gray_full.shape[0] - WIN_SIZE[1], STEP_SIZE):
                for x in range(0, gray_full.shape[1] - WIN_SIZE[0], STEP_SIZE):
                    patch = gray_full[y:y+WIN_SIZE[1], x:x+WIN_SIZE[0]]
                    feat = hog.compute(patch).reshape(1, -1)
                    score = clf.decision_function(feat)[0]
                    detections.append({"bbox": [x, y, WIN_SIZE[0], WIN_SIZE[1]], "score": score})

            if DEBUG:
                print(f"Frame {frame_idx}: {len(detections)} detections, {len(gts)} GT")

            TP_scores, FP_scores, FN_count = match_detections(detections, gts, IOU_POSITIVE)
            all_TP.extend(TP_scores)
            all_FP.extend(FP_scores)
            total_FN += FN_count

            frame_idx += 1
        cap.release()

    # ---------------- METRICS ----------------
    y_scores = np.array(all_TP + all_FP)
    y_true   = np.array([1]*len(all_TP) + [0]*len(all_FP))
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    # Save PR plot
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Merged Precision-Recall Curve (AUC={pr_auc:.3f})")
    plt.grid()
    plt.savefig(OUTPUT_PLOT)
    plt.close()

    # Save CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["precision", "recall", "threshold"])
        for p, r, t in zip(precision, recall, np.append(thresholds, np.nan)):
            writer.writerow([p, r, t])

    print("\nDONE.")
    print(f"AUC: {pr_auc:.4f}")
    print(f"TP total: {len(all_TP)}")
    print(f"FP total: {len(all_FP)}")
    print(f"FN total: {total_FN}")
    print("CSV saved to:", OUTPUT_CSV)
    print("Plot saved:", OUTPUT_PLOT)

if __name__ == "__main__":
    main()
