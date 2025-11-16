import sys
from pathlib import Path
import cv2
import numpy as np

#!/usr/bin/env python3
"""
find_armbod.py

Load images from a folder named "mili_med_og_uden_bond" (sibling to this script)
and display them one by one using OpenCV. Press any key to advance, 'q' to quit.
"""

# Folder containing images (sibling to this script)
IMAGES_DIR = Path(__file__).resolve().parent / "mili_med_og_uden_bond"
# Supported extensions
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")

def get_image_paths(folder: Path):
    paths = []
    for ext in EXTS:
        paths.extend(folder.glob(ext))
    return sorted(paths)

