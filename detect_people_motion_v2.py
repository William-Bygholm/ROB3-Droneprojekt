#!/usr/bin/env python3
import argparse
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import math
import time

WINDOW_W, WINDOW_H = 64, 128  # must match training

def extract_hog_from_patch(patch):
    """Resize patch to 64x128, convert to gray and extract HOG feature vector."""
    try:
        patch_resized = cv2.resize(patch, (WINDOW_W, WINDOW_H))
    except Exception:
        return None
    gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
    feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")
    return feat

def ensure_box_within(img_w, img_h, x, y, w, h):
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return x, y, w, h

def expand_box(x, y, w, h, expand_px, img_w, img_h):
    x2 = max(0, x - expand_px)
    y2 = max(0, y - expand_px)
    x3 = min(img_w, x + w + expand_px)
    y3 = min(img_h, y + h + expand_px)
    return x2, y2, x3 - x2, y3 - y2

def non_max_suppression_fast(boxes, scores=None, iou_thresh=0.3):
    """Simple NMS for boxes in format [x,y,w,h]. Returns indices to keep."""
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes).astype(float)
    x1 = boxes_arr[:,0]
    y1 = boxes_arr[:,1]
    x2 = boxes_arr[:,0] + boxes_arr[:,2]
    y2 = boxes_arr[:,1] + boxes_arr[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores) if scores is not None else np.argsort(y2)  # fallback order

    keep = []
    while order.size > 0:
        i = order[-1]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[:-1]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds]
    return keep

def main():
    parser = argparse.ArgumentParser(description="Fast motion-based person detection using HOG+RandomForest")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=True, help="Trained pipeline (.pkl) - HOG+RandomForest")
    parser.add_argument("--out", default=None, help="Optional output video path")
    parser.add_argument("--resize", type=float, default=0.5, help="Resize full frames to this factor for speed (0-1)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (>=1)")
    parser.add_argument("--min-area", type=int, default=500, help="Min contour area to consider (increase to reduce false positives)")
    parser.add_argument("--expand", type=int, default=10, help="Pixels to expand motion box before classification")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="NMS IoU threshold")
    parser.add_argument("--no-display", action="store_true", help="Do not show display window (faster)")
    parser.add_argument("--debug", action="store_true", help="Show debug info and timings")
    args = parser.parse_args()

    # Load model
    print("Loading model:", args.model)
    pipe = joblib.load(args.model)

    # Video open
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + args.video)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    disp_w = max(1, int(orig_w * args.resize))
    disp_h = max(1, int(orig_h * args.resize))

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (disp_w, disp_h))

    # Background subtractor (fast)
    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    frame_idx = 0
    t0 = time.time()
    processed = 0

    print(f"Video {orig_w}x{orig_h} -> processing at {disp_w}x{disp_h}. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % args.frame_skip != 0:
            continue

        t_frame = time.time()

        # Resize down for motion & display (faster)
        frame_small = cv2.resize(frame, (disp_w, disp_h))

        # Apply background subtraction on small frame
        fgmask = backsub.apply(frame_small)

        # Morphology to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Expand box a bit
            x_e, y_e, w_e, h_e = expand_box(x, y, w, h, args.expand, disp_w, disp_h)
            # Ensure minimum reasonable size
            if w_e < 16 or h_e < 32:
                continue
            candidate_boxes.append((x_e, y_e, w_e, h_e))

        # Optionally merge overlapping motion boxes via NMS before classification (reduces duplicate work)
        scores_dummy = [1.0]*len(candidate_boxes)
        keep_idx = non_max_suppression_fast(candidate_boxes, scores_dummy, iou_thresh=args.nms_iou)
        candidate_boxes = [candidate_boxes[i] for i in keep_idx]

        detections = []
        scores = []
        # For each candidate, run HOG+RF classifier
        for (bx, by, bw, bh) in candidate_boxes:
            # Crop region from the Resized frame, then scale to 64x128 inside extract_hog_from_patch
            patch = frame_small[by:by+bh, bx:bx+bw]
            feat = extract_hog_from_patch(patch)
            if feat is None:
                continue
            # Predict probability if available
            try:
                proba = pipe.predict_proba([feat])[0]
                score = float(proba[1])  # probability of 'person'
                pred = 1 if score >= 0.5 else 0
            except Exception:
                # fallback to predict()
                pred = int(pipe.predict([feat])[0])
                score = 1.0 if pred == 1 else 0.0

            if pred == 1:
                # Map box back to display coords (already in small/resized coords)
                detections.append((bx, by, bw, bh))
                scores.append(score)

        # NMS on detections to remove overlaps
        keep = non_max_suppression_fast(detections, scores if scores else None, iou_thresh=args.nms_iou)
        final_boxes = [detections[i] for i in keep] if keep else []

        # Draw detections on frame_small
        for (x, y, w, h) in final_boxes:
            cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # optionally put score
        if args.debug:
            # show motion mask for debugging
            stacked = np.hstack([cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY), fgmask])
            cv2.imshow("mask+frame", stacked)

        # Display and write
        if not args.no_display:
            cv2.imshow("Detections (motion + HOG+RF)", frame_small)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        if writer:
            writer.write(frame_small)

        processed += 1
        if args.debug:
            print(f"Frame {frame_idx} processed in {time.time()-t_frame:.3f}s, candidates {len(candidate_boxes)}, detections {len(final_boxes)}")

    t_total = time.time() - t0
    print(f"\nDone. Frames processed: {processed}. Time: {t_total:.1f}s. FPS(approx): {processed / max(1,t_total):.2f}")

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
