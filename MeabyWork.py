import cv2
import numpy as np

def detect_person(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Resize for faster processing
    image = cv2.resize(image, (800, 600))

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a range for skin/clothing colors (adjust as needed)
    lower_skin = np.array([0, 20, 70])    # Lower bound for skin/clothing
    upper_skin = np.array([20, 255, 255])  # Upper bound for skin/clothing

    # Create a mask for skin/clothing colors
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size, shape, and edges
    person_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Filter based on size and aspect ratio (adjust as needed)
        if (w > 20 and h > 50) and (0.1 < aspect_ratio < 1):
            # Check if the contour has enough edges (to reject smooth blobs)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 1:  # Minimum perimeter to consider
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
