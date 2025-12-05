import cv2
import joblib
import numpy as np
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
TEST_VIDEOS = [
    {"video": r"ProjektVideoer/2 mili en idiot der ligger ned.MP4",
     "annotations": r"Testing/2 mili og 1 idiot.json"},
    {"video": r"ProjektVideoer/3 mili 2 onde 1 god.MP4",
     "annotations": r"Testing/3mili 2 onde 1 god.json"},
]

MODEL_PATH = r"C:\Users\ehage\OneDrive\Skrivebord\Drone Projekt ROB3\ROB3-Droneprojekt\Person_Detector_Json+YOLO.pkl"
HOG_CACHE_DIR = r"C:\Users\ehage\OneDrive\HOG_cache"
WINDOW_STRIDE = (8, 8)
IOU_MATCH_THRESHOLD = 0.5
WIN_W, WIN_H = 128, 256
BLOCK_W, BLOCK_H = 32, 32
STRIDE_W, STRIDE_H = 16, 16
CELL_W, CELL_H = 8, 8
NBINS = 9
OUTPUT_PLOT = "precision_recall.png"

os.makedirs(HOG_CACHE_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    ann = {}
    image_id_to_frame = {img["id"]: idx + 1 for idx, img in enumerate(data["images"])}
    for obj in data["annotations"]:
        frame_id = image_id_to_frame[obj["image_id"]]
        x, y, w, h = obj["bbox"]
        ann.setdefault(frame_id, []).append((int(x), int(y), int(w), int(h)))
    return ann

def iou_box_xywh(a, b):
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    return inter / float(area_a + area_b - inter)

def compute_hog_descriptor(img):
    hog = cv2.HOGDescriptor(
        _winSize=(WIN_W, WIN_H),
        _blockSize=(BLOCK_W, BLOCK_H),
        _blockStride=(STRIDE_W, STRIDE_H),
        _cellSize=(CELL_W, CELL_H),
        _nbins=NBINS
    )
    desc = hog.compute(img)
    return desc

def compute_and_save_hog(frame_idx, gray):
    cache_file = os.path.join(HOG_CACHE_DIR, f"frame_{frame_idx:06d}.npz")
    if os.path.exists(cache_file):
        return cache_file  # already cached
    desc = compute_hog_descriptor(gray)
    np.savez_compressed(cache_file, desc=desc)
    return cache_file

def load_hog_from_cache(frame_idx):
    cache_file = os.path.join(HOG_CACHE_DIR, f"frame_{frame_idx:06d}.npz")
    data = np.load(cache_file)
    return data["desc"]

# ---------------- EVALUATION ----------------
def evaluate_video(video_path, annotation_path, clf):
    ann = load_annotations(annotation_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    all_scores = []
    all_labels = []

    frame_idx = 0
    hog_compute_start = time.time()
    print("=== Computing/loading HOG features ===")

    # Parallel HOG computation
    futures = {}
    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            futures[executor.submit(compute_and_save_hog, frame_idx, gray)] = frame_idx

    cap.release()
    hog_compute_end = time.time()
    print(f"HOG feature computation done: {hog_compute_end - hog_compute_start:.1f}s total")

    # Now do detection using cached features
    print("=== Running detection using cached HOG ===")
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    detection_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        desc = load_hog_from_cache(frame_idx)
        # classify entire frame as one big window for simplicity here
        score = clf.decision_function(desc.reshape(1, -1))[0]

        # assign labels based on IoU with all ground truths in this frame
        gt_boxes = ann.get(frame_idx, [])
        label = 0
        # here we just assume single window same size as frame
        frame_box = (0, 0, gray.shape[1], gray.shape[0])
        for gt in gt_boxes:
            if iou_box_xywh(frame_box, gt) >= IOU_MATCH_THRESHOLD:
                label = 1
                break
        all_scores.append(float(score))
        all_labels.append(int(label))

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            elapsed = time.time() - detection_start
            print(f"Processed {frame_idx}/{total_frames} frames | elapsed {elapsed:.1f}s", end="\r")

    cap.release()
    detection_end = time.time()
    print(f"\nDetection done: {detection_end - detection_start:.1f}s total")

    return np.array(all_scores), np.array(all_labels), hog_compute_end - hog_compute_start, detection_end - detection_start

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    if isinstance(model, dict):
        clf = model["classifier"]
    else:
        clf = model

    all_scores = []
    all_labels = []

    for entry in TEST_VIDEOS:
        video = entry["video"]
        ann = entry["annotations"]
        print(f"\n=== Evaluating {video} ===")
        s, l, t_hog, t_detect = evaluate_video(video, ann, clf)
        all_scores.append(s)
        all_labels.append(l)
        print(f"Time to compute HOG: {t_hog:.1f}s | Time for detection: {t_detect:.1f}s")

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # compute PR curve
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(rec, prec)
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_pred = (scores >= best_thr).astype(int)
    best_f1 = f1_score(labels, best_pred)
    tp = int(np.sum((best_pred == 1) & (labels == 1)))
    fp = int(np.sum((best_pred == 1) & (labels == 0)))
    fn = int(np.sum((best_pred == 0) & (labels == 1)))

    print("\n----- RESULTS -----")
    print(f"PR AUC   : {pr_auc:.4f}")
    print(f"Best F1  : {best_f1:.4f}")
    print(f"Best Thr : {best_thr:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print("-------------------")

    # save PR curve
    plt.figure()
    plt.plot(rec, prec, linewidth=1)
    plt.scatter([rec[best_idx]], [prec[best_idx]], s=60, label=f"best F1={best_f1:.3f} thr={best_thr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AUC={pr_auc:.4f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved PR curve to {OUTPUT_PLOT}")
