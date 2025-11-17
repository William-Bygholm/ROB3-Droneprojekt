import sys
from pathlib import Path
import cv2
import numpy as np

# Morphological operation parameters
MORPH_KERNEL_SIZE = (5, 5)
MORPH_ITERATIONS = 1

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

def process_image(img_path: Path):
    """
    Load one image, do some processing, return processed image.
    Change the processing steps below to experiment with different methods.
    """
    # Constant for cropping ratio (top half)
    CROP_RATIO = 0.5

    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load: {img_path}")
        return None
    
    # Remove bottom half of the image (keep only top half)
    height = img.shape[0]
    img = img[0:int(height * CROP_RATIO), :]  # keep rows 0 to height*CROP_RATIO, all columns
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask 1: Green/yellow colors (armband) - keep what's OUTSIDE this
    lower_green = np.array([15, 30, 30])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green_inv = cv2.bitwise_not(mask_green)  # invert to remove green/yellow
    
    # Mask 2: Orange background - remove this too
    # Orange is typically H: 10-25, S: 100-255, V: 100-255 (tune as needed)
    lower_orange = np.array([8, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_orange_inv = cv2.bitwise_not(mask_orange)  # invert to remove orange
    
    # Apply morphology to clean up mask (remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find all connected components (blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    
    # Find the largest blob (excluding background label 0)
    if num_labels > 1:
        # stats columns: [left, top, width, height, area]
        # label 0 is background, so start from label 1
        areas = stats[1:, cv2.CC_STAT_AREA]  # get areas of all blobs except background
        largest_label = np.argmax(areas) + 1  # +1 because we sliced from index 1
        
        # Create mask with only the largest blob
        largest_blob_mask = np.zeros_like(combined_mask)
        largest_blob_mask[labels == largest_label] = 255
        
        combined_mask = largest_blob_mask
    
    # Apply final mask to original image
    processed = cv2.bitwise_and(img, img, mask=combined_mask)
    
    # Optional: show the combined mask itself (for debugging)
    # processed = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    
    return processed

def display_images():
    """
    Load images one by one, process, and display.
    Press 'n' or space to go to next, 'b' to go back, 'q' or ESC to quit.
    """
    image_paths = get_image_paths(IMAGES_DIR)
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
    idx = 0
    window_name = "Processed Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    consecutive_failures = 0
    max_failures = 10  # You can adjust this threshold as needed
    
    while True:
        img_path = image_paths[idx]
        processed = process_image(img_path)
        
        if processed is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"Too many consecutive image load failures ({max_failures}). Exiting.")
                break
            idx = min(idx + 1, len(image_paths) - 1)
            continue
        else:
            consecutive_failures = 0
        
        # Add text overlay with filename and index
        display = processed.copy()
        #text = f"{idx+1}/{len(image_paths)}: {img_path.name}"
        #cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
        #            0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break
        elif key == ord('n') or key == ord(' '):  # n or space to next
            idx = min(idx + 1, len(image_paths) - 1)
        elif key == ord('b'):  # b to go back
            idx = max(idx - 1, 0)
    
    cv2.destroyAllWindows()
        elif key == ord('b'):  # b to go back
            idx = max(idx - 1, 0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_images()

