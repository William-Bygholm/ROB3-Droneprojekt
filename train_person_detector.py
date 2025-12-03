import cv2
import numpy as np
import json
import os
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------- CONFIG ---------------- #
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

ADDITIONAL_POSITIVES_DIR = 0
ADDITIONAL_NEGATIVES_DIR = 0

OUTPUT_MODEL = "Person_Detector_Json.pkl"
WINDOW_SIZE = (128, 256)
NEGATIVE_SAMPLES_PER_FRAME = 5
MAX_FRAME_WIDTH = 1280   # downscale hvis video er meget stor
MAX_FRAME_HEIGHT = 720

# ---------------- TRAINER CLASS ---------------- #
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

    # ---------- LOAD ANNOTATIONS ---------- #
    def load_annotations(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        frame_annotations = {}
        for ann in data.get('annotations', []):
            image_id = ann['image_id']
            frame_annotations.setdefault(image_id, []).append(ann)
        return frame_annotations, data.get('images', [])

    # ---------- SAMPLE EXTRACTION ---------- #
    def extract_positive_samples(self, frame, annotations):
        samples = []
        for ann in annotations:
            x, y, w, h = [int(v) for v in ann['bbox']]
            roi = frame[y:y+h, x:x+w]
            if roi.shape[0] < 32 or roi.shape[1] < 32:
                continue
            try:
                resized = cv2.resize(roi, self.window_size)
                samples.append(resized)
            except cv2.error:
                continue
        return samples

    def extract_negative_samples(self, frame, annotations, num_samples=5):
        samples = []
        h, w = frame.shape[:2]
        positive_boxes = [(int(a['bbox'][0]), int(a['bbox'][1]),
                           int(a['bbox'][0]+a['bbox'][2]), int(a['bbox'][1]+a['bbox'][3]))
                          for a in annotations]

        attempts = 0
        max_attempts = num_samples * 10
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            x = np.random.randint(0, max(1, w - self.window_size[0]))
            y = np.random.randint(0, max(1, h - self.window_size[1]))
            candidate_box = (x, y, x+self.window_size[0], y+self.window_size[1])

            if any(not (candidate_box[2] < px1 or candidate_box[0] > px2 or
                        candidate_box[3] < py1 or candidate_box[1] > py2)
                   for px1, py1, px2, py2 in positive_boxes):
                continue

            roi = frame[y:y+self.window_size[1], x:x+self.window_size[0]]
            if roi.shape[:2] == (self.window_size[1], self.window_size[0]):
                samples.append(roi)
        return samples

    def extract_features_from_samples(self, samples):
        features = []
        for sample in samples:
            if len(sample.shape) == 3:
                sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            hog_features = self.hog.compute(sample)
            features.append(hog_features.ravel())
        return np.array(features)

    # ---------- COLLECT DATA FROM VIDEO ---------- #
    def collect_training_data(self, video_path, frame_annotations, sample_every_n_frames=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        # downscale hvis video er stor
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAX_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAX_FRAME_HEIGHT)

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_idx += 1
                if frame_idx >= total_frames:
                    break
                continue

            if frame_idx % sample_every_n_frames != 0:
                frame_idx += 1
                continue

            matching_annotations = frame_annotations.get(frame_idx, [])
            if matching_annotations:
                self.positive_samples.extend(self.extract_positive_samples(frame, matching_annotations))
                self.negative_samples.extend(self.extract_negative_samples(frame, matching_annotations, NEGATIVE_SAMPLES_PER_FRAME))

            frame_idx += 1
            if frame_idx >= total_frames:
                break

        cap.release()
        print(f"[INFO] Data collection finished. Positives: {len(self.positive_samples)}, Negatives: {len(self.negative_samples)}")

    # ---------- TRAIN SVM ---------- #
    def train_svm(self, test_size=0.2, C=0.1):
        print("[INFO] Extracting HOG features...")
        X_pos = self.extract_features_from_samples(self.positive_samples)
        X_neg = self.extract_features_from_samples(self.negative_samples)
        y_pos = np.ones(len(X_pos))
        y_neg = np.zeros(len(X_neg))

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

        print(f"[INFO] Training with {len(X)} samples...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        clf = LinearSVC(C=C, max_iter=10000, random_state=42, verbose=1)
        clf.fit(X_train, y_train)

        print("\n=== TRAIN SET ===")
        print(classification_report(y_train, clf.predict(X_train)))
        print("\n=== TEST SET ===")
        print(classification_report(y_test, clf.predict(X_test)))
        print("Confusion Matrix (Test):")
        print(confusion_matrix(y_test, clf.predict(X_test)))

        return clf

    def save_model(self, clf, output_path):
        joblib.dump({'classifier': clf}, output_path)
        print(f"[INFO] Model saved to: {output_path}")

# ---------------- MAIN ---------------- #
def main():
    trainer = PersonDetectorTrainer(window_size=WINDOW_SIZE)

    for entry in TRAINING_DATA:
        video_path = entry["video"]
        json_path = entry["json"]
        if not os.path.exists(video_path) or not os.path.exists(json_path):
            print(f"[WARNING] Missing video or annotation, skipping: {video_path}")
            continue
        frame_annotations, _ = trainer.load_annotations(json_path)
        trainer.collect_training_data(video_path, frame_annotations)

    #if ADDITIONAL_POSITIVES_DIR and os.path.exists(ADDITIONAL_POSITIVES_DIR):
    #    trainer.load_positive_images_from_folder(ADDITIONAL_POSITIVES_DIR)
    #if ADDITIONAL_NEGATIVES_DIR and os.path.exists(ADDITIONAL_NEGATIVES_DIR):
    #    trainer.load_negative_images_from_folder(ADDITIONAL_NEGATIVES_DIR)

    print(f"[INFO] Total positives: {len(trainer.positive_samples)}, Total negatives: {len(trainer.negative_samples)}")
    if len(trainer.positive_samples) < 5:
        print("[ERROR] Not enough positive samples, aborting.")
        return

    clf = trainer.train_svm(test_size=0.2, C=0.1)
    trainer.save_model(clf, OUTPUT_MODEL)
    print("[INFO] Training complete!")

if __name__ == "__main__":
    main()
