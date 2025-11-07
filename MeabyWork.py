import cv2
import numpy as np

def detect_person(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Resize for faster processing
    image = cv2.resize(image, (640, 640))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to isolate foreground
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 1
    )

    # Morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and shape
    person_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Filter based on size and aspect ratio (adjust as needed)
        if (w > 25 and h > 50) and (0.3 < aspect_ratio < 1.0):
            person_contours.append(contour)

    # Draw bounding boxes around detected persons
    for contour in person_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Person Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_person("Billeder/IMG_20251104_094347.jpg")  # Replace with your image path
