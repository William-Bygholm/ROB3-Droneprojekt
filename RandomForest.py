import argparse
import glob
import os
from pathlib import Path
import joblib
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


def extract_hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Extract HOG features from an image."""
    if img is None:
        return None
    
    img = cv2.resize(img, (64, 128))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
    )


def load_dataset(pos_dir, neg_dir):
    X, y = [], []

    def load_images(directory, label):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            files.extend(glob.glob(os.path.join(directory, ext)))

        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            feat = extract_hog(img)
            if feat is not None:
                X.append(feat)
                y.append(label)

    load_images(pos_dir, 1)
    load_images(neg_dir, 0)

    if len(X) == 0:
        raise RuntimeError("No images found in positive or negative folders!")

    return np.array(X), np.array(y)


def build_pipeline(n_estimators=200, n_jobs=-1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=42
        )),
    ])


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest with HOG features")
    parser.add_argument("--pos", required=True, help="Folder with positive images")
    parser.add_argument("--neg", required=True, help="Folder with negative images")
    parser.add_argument("--out", default="hog_rf_pipeline.pkl", help="Output model file")
    parser.add_argument("--estimators", type=int, default=200, help="Number of trees")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    args = parser.parse_args()

    pos_dir = Path("C:/Users/alexa/Downloads/positives1/positives")
    neg_dir = Path("C:/Users/alexa/Downloads/negatives1/negatives")

    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError("Positive or negative directory does NOT exist!")

    print("Loading dataset...")
    X, y = load_dataset(str(pos_dir), str(neg_dir))
    print(f"Loaded {len(X)} samples: {y.sum()} positives, {len(y)-y.sum()} negatives")

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    print("Training RandomForest model...")
    pipe = build_pipeline(n_estimators=args.estimators)
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    try:
        print("Cross-validation...")
        cv_scores = cross_val_score(pipe, X, y, cv=3, n_jobs=-1)
        print("CV Scores:", cv_scores, "Mean:", np.mean(cv_scores))
    except Exception as e:
        print("Cross-validation failed:", e)

    joblib.dump(pipe, args.out)
    print("Model saved to:", args.out)


if __name__ == "__main__":
    main()
