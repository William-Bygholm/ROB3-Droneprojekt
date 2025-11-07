import cv2
import numpy as np

def detect_fire_and_edges(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Resize the image (optional)
    image = cv2.resize(image, (800, 600))

    # --- Grass Fire Detection (Color-Based Blob Detection) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define fire color range (red, orange, yellow)
    lower_fire = np.array([0, 50, 50])
    upper_fire = np.array([20, 255, 255])

    # Create a mask for fire-like colors
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Apply Gaussian blur to reduce noise
    blurred_fire = cv2.GaussianBlur(fire_mask, (5, 5), 0)

    # Set up SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100    # Adjust as needed
    params.maxArea = 10000   # Adjust as needed

    # Create a detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect fire blobs
    fire_keypoints = detector.detect(blurred_fire)

    # Draw fire blobs on the original image
    fire_image = cv2.drawKeypoints(
        image.copy(),
        fire_keypoints,
        np.array([]),
        (0, 0, 255),  # Red color for fire blobs
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # --- Sobel Edge Detection on Fire Regions ---
    # Convert fire mask to grayscale (already is, but just in case)
    fire_mask_gray = fire_mask.copy()

    # Apply Sobel edge detection to the fire mask
    sobelx = cv2.Sobel(fire_mask_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(fire_mask_gray, cv2.CV_64F, 0, 1, ksize=5)

    # Combine Sobel results
    sobel_combined = cv2.addWeighted(
        cv2.convertScaleAbs(sobelx), 0.5,
        cv2.convertScaleAbs(sobely), 0.5,
        0
    )

    # Apply the Sobel edges to the original image (highlight fire edges)
    edges_image = image.copy()
    edges_image[sobel_combined > 50] = [0, 255, 0]  # Green edges for fire boundaries

    # --- Combine Results ---
    # Overlay fire blobs and edges on the original image
    final_image = edges_image.copy()
    final_image = cv2.drawKeypoints(
        final_image,
        fire_keypoints,
        np.array([]),
        (0, 0, 255),  # Red color for fire blobs
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the results
    cv2.imshow("Fire Detection + Sobel Edges", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_fire_and_edges("Billeder/IMG_20251104_094347.jpg")  # Replace with your image path
