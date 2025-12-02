"""
Train a person detector using manually annotated COCO data.
Supports both HOG+SVM and deep learning approaches.
"""

import cv2
import numpy as np
import json
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

# Configuration - Multiple videos and annotations
TRAINING_DATA = [
    {
        'video': r"ProjektVideoer/2 mili der ligger ned og 1 civil.MP4",
        'json': r"Træning/2 mili der ligger ned og 1 civil.json"
    },
    {
        'video': r"ProjektVideoer/Civil person.MP4",
        'json': r"Træning/Civil person.json"
    }
]
# Additional positive samples from external folder
ADDITIONAL_POSITIVES_DIR = r"D:\Pos"  # Set to None to skip

OUTPUT_MODEL = "person_detector_trained.pkl"
WINDOW_SIZE = (128, 256)  # Standard HOG window size for person detection
NEGATIVE_SAMPLES_PER_FRAME = 5  # How many negative samples to extract per frame

class PersonDetectorTrainer:
    def __init__(self, window_size=(128, 256)):
        self.window_size = window_size
        self.hog = cv2.HOGDescriptor(
            _winSize=window_size,
            _blockSize=(32, 32),
            _blockStride=(16, 16),
            _cellSize=(8, 8),
            _nbins=9
        )
        self.positive_samples = []
        self.negative_samples = []
        
    def load_annotations(self, json_path):
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
    
    def extract_positive_samples(self, frame, annotations):
        """Extract positive samples (people) from annotated bounding boxes"""
        samples = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = [int(v) for v in bbox]
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Skip if too small
            if roi.shape[0] < 32 or roi.shape[1] < 32:
                continue
            
            # Resize to standard window size
            try:
                resized = cv2.resize(roi, self.window_size)
                samples.append(resized)
            except:
                continue
        
        return samples
    
    def extract_negative_samples(self, frame, annotations, num_samples=5):
        """Extract negative samples (non-people) from frame"""
        samples = []
        h, w = frame.shape[:2]
        
        # Get all positive bounding boxes
        positive_boxes = []
        for ann in annotations:
            bbox = ann['bbox']
            x, y, ww, hh = [int(v) for v in bbox]
            positive_boxes.append((x, y, x+ww, y+hh))
        
        # Try to extract random patches that don't overlap with people
        attempts = 0
        max_attempts = num_samples * 10
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Random position
            x = np.random.randint(0, max(1, w - self.window_size[0]))
            y = np.random.randint(0, max(1, h - self.window_size[1]))
            
            # Check if overlaps with any positive box
            candidate_box = (x, y, x + self.window_size[0], y + self.window_size[1])
            
            overlaps = False
            for px1, py1, px2, py2 in positive_boxes:
                # Check intersection
                if not (candidate_box[2] < px1 or candidate_box[0] > px2 or
                       candidate_box[3] < py1 or candidate_box[1] > py2):
                    overlaps = True
                    break
            
            if not overlaps:
                roi = frame[y:y+self.window_size[1], x:x+self.window_size[0]]
                if roi.shape[:2] == (self.window_size[1], self.window_size[0]):
                    samples.append(roi)
        
        return samples
    
    def extract_features_from_samples(self, samples):
        """Extract HOG features from image samples"""
        features = []
        for sample in samples:
            # Convert to grayscale if needed
            if len(sample.shape) == 3:
                sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            
            # Compute HOG features
            hog_features = self.hog.compute(sample)
            features.append(hog_features.ravel())
        
        return np.array(features)
    
    def load_positive_images_from_folder(self, folder_path):
        """Load positive samples from a folder of images"""
        if not os.path.exists(folder_path):
            print(f"WARNING: Folder not found: {folder_path}")
            return
        
        print(f"\nLoading positive samples from: {folder_path}")
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        loaded_count = 0
        skipped_count = 0
        
        # Walk through all files in the folder and subfolders
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in extensions:
                    continue
                
                file_path = os.path.join(root, filename)
                
                # Load image
                img = cv2.imread(file_path)
                if img is None:
                    skipped_count += 1
                    continue
                
                # Resize to standard window size
                try:
                    resized = cv2.resize(img, self.window_size)
                    self.positive_samples.append(resized)
                    loaded_count += 1
                    
                    if loaded_count % 100 == 0:
                        print(f"Loaded {loaded_count} images...")
                        
                except Exception as e:
                    skipped_count += 1
                    continue
        
        print(f"Loaded {loaded_count} positive samples from folder")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files (invalid or unreadable)")
    
    def collect_training_data(self, video_path, frame_annotations, sample_every_n_frames=1):
        """Collect positive and negative samples from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_with_annotations = 0
        
        # Find the annotation ID range to determine offset
        if frame_annotations:
            min_id = min(frame_annotations.keys())
            max_id = max(frame_annotations.keys())
            print(f"Annotation IDs range: {min_id} to {max_id}")
            print(f"Total frames in video: {total_frames}")
            print(f"Annotated frames available: {len(frame_annotations)}")
        
        print(f"\nCollecting training data from all frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame to speed up (default=1 means all frames)
            if frame_idx % sample_every_n_frames != 0:
                frame_idx += 1
                continue
            
            # Find annotations for this frame (try different offsets)
            matching_annotations = []
            # Try direct match and common offsets
            for possible_id in [frame_idx, frame_idx + 1, frame_idx - 1, 
                               3720 + frame_idx, 3726 + frame_idx, 
                               min_id + frame_idx]:
                if possible_id in frame_annotations:
                    matching_annotations = frame_annotations[possible_id]
                    frames_with_annotations += 1
                    break
            
            if matching_annotations:
                # Extract positive samples (people)
                positives = self.extract_positive_samples(frame, matching_annotations)
                self.positive_samples.extend(positives)
                
                # Extract negative samples (background)
                negatives = self.extract_negative_samples(frame, matching_annotations, NEGATIVE_SAMPLES_PER_FRAME)
                self.negative_samples.extend(negatives)
                
                if frame_idx % 50 == 0:
                    print(f"Frame {frame_idx}/{total_frames}: "
                          f"Matched {frames_with_annotations} frames, "
                          f"Positives: {len(self.positive_samples)}, "
                          f"Negatives: {len(self.negative_samples)}")
            
            frame_idx += 1
        
        cap.release()
        print(f"\nData collection complete!")
        print(f"Total positive samples: {len(self.positive_samples)}")
        print(f"Total negative samples: {len(self.negative_samples)}")
    
    def train_svm(self, test_size=0.2, C=0.1):
        """Train SVM classifier"""
        print("\nExtracting HOG features...")
        
        # Extract features
        X_pos = self.extract_features_from_samples(self.positive_samples)
        X_neg = self.extract_features_from_samples(self.negative_samples)
        
        # Create labels
        y_pos = np.ones(len(X_pos))
        y_neg = np.zeros(len(X_neg))
        
        # Combine
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])
        
        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train SVM
        print(f"\nTraining SVM (C={C})...")
        clf = LinearSVC(C=C, max_iter=10000, random_state=42, verbose=1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        print("\n=== Training Set Performance ===")
        y_pred_train = clf.predict(X_train)
        print(classification_report(y_train, y_pred_train, target_names=['Background', 'Person']))
        
        print("\n=== Test Set Performance ===")
        y_pred_test = clf.predict(X_test)
        print(classification_report(y_test, y_pred_test, target_names=['Background', 'Person']))
        
        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_pred_test)
        print(cm)
        print("[[TN FP]")
        print(" [FN TP]]")
        
        return clf
    
    def save_model(self, clf, output_path):
        """Save trained model"""
        model_data = {
            'classifier': clf,
            'hog_params': {
                'winSize': self.window_size,
                'blockSize': (16, 16),
                'blockStride': (8, 8),
                'cellSize': (8, 8),
                'nbins': 9
            }
        }
        joblib.dump(model_data, output_path)
        print(f"\nModel saved to: {output_path}")

def main():
    # Initialize trainer
    trainer = PersonDetectorTrainer(window_size=WINDOW_SIZE)
    
    # Process each video/annotation pair
    for idx, data_pair in enumerate(TRAINING_DATA, 1):
        video_path = data_pair['video']
        json_path = data_pair['json']
        
        print(f"\n{'='*60}")
        print(f"Processing dataset {idx}/{len(TRAINING_DATA)}")
        print(f"Video: {video_path}")
        print(f"Annotations: {json_path}")
        print('='*60)
        
        # Check if files exist
        if not os.path.exists(video_path):
            print(f"WARNING: Video not found: {video_path}")
            continue
        if not os.path.exists(json_path):
            print(f"WARNING: Annotation file not found: {json_path}")
            continue
        
        # Load annotations for this video
        frame_annotations, images = trainer.load_annotations(json_path)
        print(f"Loaded {len(frame_annotations)} annotated frames")
        
        # Collect training data from this video (use ALL frames)
        trainer.collect_training_data(video_path, frame_annotations, sample_every_n_frames=1)
    
    # Load additional positive samples from external folder
    if ADDITIONAL_POSITIVES_DIR and os.path.exists(ADDITIONAL_POSITIVES_DIR):
        print(f"\n{'='*60}")
        print("Loading additional positive samples from external folder")
        print('='*60)
        trainer.load_positive_images_from_folder(ADDITIONAL_POSITIVES_DIR)
    
    # Check if we have enough data across all videos
    print(f"\n{'='*60}")
    print("TOTAL DATA COLLECTED:")
    print(f"Positive samples: {len(trainer.positive_samples)}")
    print(f"Negative samples: {len(trainer.negative_samples)}")
    print('='*60)
    
    if len(trainer.positive_samples) < 10:
        print("\nERROR: Not enough positive samples! Check frame ID mapping.")
        return
    
    # Train model on combined data from all videos
    clf = trainer.train_svm(test_size=0.2, C=0.01)
    
    # Save model
    trainer.save_model(clf, OUTPUT_MODEL)
    
    print("\n=== Training Complete! ===")
    print(f"You can now use '{OUTPUT_MODEL}' for person detection")
    print("Use it in your detection script by loading:")
    print(f"  model = joblib.load('{OUTPUT_MODEL}')")

if __name__ == "__main__":
    main()
