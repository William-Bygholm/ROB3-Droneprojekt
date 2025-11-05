import cv2
import numpy as np
from pathlib import Path

# Define the folder path
folder_path = Path("Billeder")

# Check if the folder exists
if not folder_path.exists():
    raise FileNotFoundError(f"The folder {folder_path} does not exist.")

# List all image files in the folder
image_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
if not image_files:
    raise FileNotFoundError(f"No image files found in the folder {folder_path}.")
print(f"Found {len(image_files)} image files.")
# Process each image file
for image_file in image_files:
    # Read the image
    image = cv2.imread(str(image_file))
    if image is None:
        print(f"Failed to read image {image_file}. Skipping.")
        continue

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Resize Original Images
    resized_image = cv2.resize(image, None, fx=0.1, fy=0.1)

    # Resize Edges Original Images
    resized_image_Edges = cv2.resize(edges, None, fx=0.1, fy=0.1)

    # Display the original and processed images
    cv2.imshow("Original Image", resized_image)
    cv2.imshow("Edges", resized_image_Edges)
    cv2.waitKey(0)

