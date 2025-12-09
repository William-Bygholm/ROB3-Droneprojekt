"""
Debug Visualization Script
Shows GT boxes (green) and Detections (red) on the same frame
"""

import cv2
import json
import numpy as np
from Main import clf, hog, detect_people, FRAME_SKIP
from test_100_frames import load_coco_annotations, extract_frame_number, compute_iou

# Config
VIDEO_PATH = "ProjektVideoer/2 mili en idiot der ligger ned.MP4"
JSON_PATH = "Testing/2 mili og 1 idiot.json"
FRAME_TO_CHECK = 2  # First processed frame with FRAME_SKIP=2
IOU_THRESHOLD = 0.3

print("=" * 80)
print("DEBUG VISUALIZATION - GT vs Detections")
print("=" * 80)

# 1. Check JSON structure
print("\n[1] Checking JSON structure...")
with open(JSON_PATH, 'r') as f:
    coco = json.load(f)
    
print(f"Total images in JSON: {len(coco['images'])}")
print(f"Total annotations in JSON: {len(coco['annotations'])}")
print(f"\nFirst 5 image filenames:")
for i, img in enumerate(coco['images'][:5]):
    frame_num = extract_frame_number(img['file_name'])
    print(f"  {img['file_name']} → Frame {frame_num}")

# 2. Load GT annotations
print(f"\n[2] Loading GT annotations...")
gt_frames = load_coco_annotations(JSON_PATH)
print(f"Total frames with GT: {len(gt_frames)}")
print(f"Frame numbers with GT (first 10): {list(gt_frames.keys())[:10]}")

# 3. Open video and read frame
print(f"\n[3] Reading frame {FRAME_TO_CHECK} from video...")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total video frames: {total_frames}")

# Simulate the same frame reading logic as test script
frame_id = 0
target_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    if frame_id == FRAME_TO_CHECK:
        target_frame = frame.copy()
        break

cap.release()

if target_frame is None:
    print(f"ERROR: Could not read frame {FRAME_TO_CHECK}")
    exit(1)

print(f"Frame {FRAME_TO_CHECK} loaded: {target_frame.shape}")

# 4. Get GT boxes for this frame
gt_boxes = gt_frames.get(FRAME_TO_CHECK, [])
print(f"\n[4] Ground Truth for frame {FRAME_TO_CHECK}:")
print(f"GT boxes found: {len(gt_boxes)}")
if len(gt_boxes) > 0:
    for i, gt in enumerate(gt_boxes):
        print(f"  GT {i}: {gt['bbox']} - {gt['label']}")
else:
    print("  WARNING: No GT boxes for this frame!")

# 5. Run detector
print(f"\n[5] Running detector on frame {FRAME_TO_CHECK}...")
detections = detect_people(target_frame, clf, hog)
print(f"Detections found: {len(detections)}")
if len(detections) > 0:
    for i, det in enumerate(detections[:5]):  # Show first 5
        print(f"  Detection {i}: {det}")

# 6. Calculate IoU for all pairs
print(f"\n[6] IoU Matrix (threshold = {IOU_THRESHOLD}):")
if len(gt_boxes) > 0 and len(detections) > 0:
    print(f"{'':15} ", end="")
    for i in range(len(detections)):
        print(f"Det{i:2d}  ", end="")
    print()
    
    for j, gt in enumerate(gt_boxes):
        print(f"GT{j} ({gt['label'][:10]:10s})", end=" ")
        for i, det in enumerate(detections):
            iou = compute_iou(det, gt['bbox'])
            if iou >= IOU_THRESHOLD:
                print(f"\033[92m{iou:5.3f}\033[0m ", end="")  # Green if match
            else:
                print(f"{iou:5.3f} ", end="")
        print()
    
    # Count matches
    matches = 0
    for gt in gt_boxes:
        for det in detections:
            if compute_iou(det, gt['bbox']) >= IOU_THRESHOLD:
                matches += 1
                break
    
    print(f"\nPotential matches (IoU >= {IOU_THRESHOLD}): {matches}/{len(gt_boxes)} GT boxes")
else:
    print("Cannot compute IoU - missing GT or detections")

# 7. Visualize
print(f"\n[7] Creating visualization...")
vis_frame = target_frame.copy()

# Draw GT boxes in GREEN
for i, gt in enumerate(gt_boxes):
    x1, y1, x2, y2 = gt['bbox']
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"GT: {gt['label']}"
    cv2.putText(vis_frame, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Draw Detections in RED
for i, det in enumerate(detections):
    x1, y1, x2, y2 = [int(v) for v in det]
    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label = f"Det {i}"
    cv2.putText(vis_frame, label, (x1, y2 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Add legend
legend_y = 30
cv2.putText(vis_frame, f"Frame {FRAME_TO_CHECK}", (10, legend_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(vis_frame, f"GREEN = GT ({len(gt_boxes)} boxes)", (10, legend_y + 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(vis_frame, f"RED = Detections ({len(detections)} boxes)", (10, legend_y + 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save and show
output_path = "Output/debug_frame.png"
cv2.imwrite(output_path, vis_frame)
print(f"Visualization saved to: {output_path}")

# Show window
cv2.imshow(f"Debug - Frame {FRAME_TO_CHECK}", vis_frame)
print("\nPress any key in the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)

if len(gt_boxes) == 0:
    print("❌ NO GT BOXES FOUND!")
    print("   → Frame number mismatch problem")
    print("   → Check if annotations are on different frame numbers")
elif len(detections) == 0:
    print("❌ NO DETECTIONS FOUND!")
    print("   → Detector failed completely on this frame")
elif matches == 0:
    print("❌ NO MATCHES (IoU < 0.3)!")
    print("   → Detections and GT boxes don't overlap")
    print("   → Check visual - are boxes in completely different locations?")
elif matches < len(gt_boxes):
    print(f"⚠️  PARTIAL MATCHES: {matches}/{len(gt_boxes)}")
    print("   → Some GT boxes not detected (False Negatives)")
    print("   → Detector performance issue")
else:
    print(f"✅ ALL GT BOXES MATCHED: {matches}/{len(gt_boxes)}")
    print("   → Logic is working correctly")
    print("   → Poor overall results are due to detector performance")

print("=" * 80)
