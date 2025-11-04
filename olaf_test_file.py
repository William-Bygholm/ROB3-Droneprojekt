import cv2
import numpy as np
from pathlib import Path

# Define the folder path
folder_path = Path(r"C:\Users\olafa\Documents\GitHub\ROB3-Droneprojekt\Billeder")

# Check if the folder exists
if not folder_path.exists():
    raise FileNotFoundError(f"The folder {folder_path} does not exist.")
