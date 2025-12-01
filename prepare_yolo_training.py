"""
Prepare COCO annotations for YOLO training format.
Extracts frames and converts bounding boxes to YOLO format.
"""

import cv2
import json
import os
from pathlib import Path

# Configuration
VIDEO_PATH = r"ProjektVideoer/3 mili 2 onde 1 god.MP4"
COCO_JSON = "instances_default.json"
OUTPUT_DIR = "yolo_dataset"
TRAIN_SPLIT = 0.8  # 80% train, 20% validation

def load_annotations(json_path):
    """Load COCO annotations"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Organize by image_id
    frame_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in frame_annotations:
            frame_annotations[image_id] = []
        frame_annotations[image_id].append(ann)
    
    return frame_annotations, data['images']

def coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All normalized by image dimensions (0-1)
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def extract_frames_and_labels(video_path, frame_annotations, output_dir):
    """
    Extract annotated frames from video and save with YOLO labels
    """
    # Create directories
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_idx = 0
    saved_count = 0
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video dimensions: {img_width}x{img_height}")
    print("Extracting annotated frames...")
    
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Find annotations for this frame (try different offsets)
        matching_annotations = []
        matched_id = None
        for possible_id in [frame_idx, frame_idx + 1, 3720 + frame_idx, 3726 + frame_idx]:
            if possible_id in frame_annotations:
                matching_annotations = frame_annotations[possible_id]
                matched_id = possible_id
                break
        
        if matching_annotations:
            # Save frame
            frame_filename = f"frame_{saved_count:05d}.jpg"
            frame_path = images_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Save YOLO labels
            label_filename = f"frame_{saved_count:05d}.txt"
            label_path = labels_dir / label_filename
            
            with open(label_path, 'w') as f:
                for ann in matching_annotations:
                    bbox = ann['bbox']  # [x, y, width, height]
                    
                    # Convert to YOLO format
                    x_center, y_center, width, height = coco_to_yolo_bbox(
                        bbox, img_width, img_height
                    )
                    
                    # Class 0 = person (single class detection)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            extracted_frames.append(frame_filename)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"Extracted {saved_count} frames (video frame {frame_idx})...")
        
        frame_idx += 1
    
    cap.release()
    print(f"\nExtracted {saved_count} annotated frames")
    return extracted_frames

def create_train_val_split(frame_list, output_dir, train_split=0.8):
    """Create train/val split files"""
    import random
    random.seed(42)
    
    shuffled = frame_list.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_split)
    train_files = shuffled[:split_idx]
    val_files = shuffled[split_idx:]
    
    # Write train.txt
    with open(Path(output_dir) / "train.txt", 'w') as f:
        for filename in train_files:
            f.write(f"./images/{filename}\n")
    
    # Write val.txt
    with open(Path(output_dir) / "val.txt", 'w') as f:
        for filename in val_files:
            f.write(f"./images/{filename}\n")
    
    print(f"Train: {len(train_files)} images")
    print(f"Val: {len(val_files)} images")

def create_yaml_config(output_dir):
    """Create dataset.yaml for YOLO training"""
    yaml_content = f"""# Person detection dataset
path: {os.path.abspath(output_dir)}  # dataset root dir
train: train.txt  # train images (relative to 'path')
val: val.txt  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['person']  # class names
"""
    
    with open(Path(output_dir) / "dataset.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated dataset.yaml configuration")

def main():
    print("=== YOLO Dataset Preparation ===\n")
    
    # Load annotations
    print("Loading annotations...")
    frame_annotations, images = load_annotations(COCO_JSON)
    print(f"Loaded {len(frame_annotations)} annotated frames")
    
    # Extract frames and convert labels
    extracted_frames = extract_frames_and_labels(VIDEO_PATH, frame_annotations, OUTPUT_DIR)
    
    if len(extracted_frames) == 0:
        print("\nERROR: No frames extracted! Check frame ID mapping in extract_frames_and_labels()")
        return
    
    # Create train/val split
    create_train_val_split(extracted_frames, OUTPUT_DIR, TRAIN_SPLIT)
    
    # Create YAML config
    create_yaml_config(OUTPUT_DIR)
    
    print("\n=== Dataset Ready! ===")
    print(f"Dataset location: {os.path.abspath(OUTPUT_DIR)}")
    print("\nTo train YOLOv8:")
    print("1. Install: pip install ultralytics")
    print("2. Train: yolo detect train data=yolo_dataset/dataset.yaml model=yolov8n.pt epochs=50 imgsz=640")
    print("\nFor YOLOv5:")
    print("1. Clone: git clone https://github.com/ultralytics/yolov5")
    print("2. Train: python train.py --data ../yolo_dataset/dataset.yaml --weights yolov5s.pt --epochs 50")

if __name__ == "__main__":
    main()
