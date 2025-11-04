import cv2
import numpy as np
from pathlib import Path

# Define the folder path
folder_path = Path("Billeder")

# Check if the folder exists
if not folder_path.exists():
    raise FileNotFoundError(f"The folder {folder_path} does not exist.")
