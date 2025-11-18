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
    """Extract HOG features from a grayscale image (expects uint8 or float image).

    Returns a 1D numpy array of HOG features.
    """
    if img is None:
        return None
    # Ensure proper size and type
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


def load_dataset(pos_dir, neg_dir, pattern="*.png"):
    X = []
    y = []

    def add_from_dir(d, label):
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            files.extend(glob.glob(os.path.join(d, ext)))
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            feat = extract_hog(img)
            if feat is None:
                continue
            X.append(feat)
            y.append(label)

    add_from_dir(pos_dir, 1)
    add_from_dir(neg_dir, 0)

    if len(X) == 0:
        raise RuntimeError(f"No images found in {pos_dir} or {neg_dir}.")

    return np.array(X), np.array(y)


def build_pipeline(n_estimators=200, n_jobs=-1):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42),
        ),
    ])
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Train a RandomForest on HOG features")
    parser.add_argument("--pos", required=True, help="Directory with positive images (people)")
    parser.add_argument("--neg", required=True, help="Directory with negative images")
    parser.add_argument("--out", default="hog_rf_pipeline.pkl", help="Output model path (joblib)")
    parser.add_argument("--estimators", type=int, default=200, help="Number of trees")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    args = parser.parse_args()

    pos_dir = Path(args.pos)
    neg_dir = Path(args.neg)
    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError("Positive or negative directory does not exist")

    print("Loading dataset...")
    X, y = load_dataset(str(pos_dir), str(neg_dir))
    print(f"Loaded {len(X)} samples (positives={y.sum()}, negatives={len(y)-y.sum()})")

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    print("Building pipeline and training...")
    pipe = build_pipeline(n_estimators=args.estimators)
    pipe.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    # cross-validation (quick)
    try:
        print("Cross-validating (3-fold)...")
        cv_scores = cross_val_score(pipe, X, y, cv=3, n_jobs=-1)
        print("CV scores:", cv_scores, "mean:", np.mean(cv_scores))
    except Exception as e:
        print("Cross-validation failed:", e)

    out_path = args.out
    joblib.dump(pipe, out_path)
    print(f"Model pipeline saved -> {out_path}")


if __name__ == "__main__":
    main()

