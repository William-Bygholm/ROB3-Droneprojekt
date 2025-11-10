import cv2
import numpy as np

# ----------------------------
# Function: detect military jacket
# ----------------------------
def detect_military_jacket(image, bbox):
    x, y, w, h = bbox

    # Crop torso region (middle of person)
    torso = image[y + int(h * 0.25): y + int(h * 0.60), x:x + w]
    if torso.size == 0:
        return False

    # Convert to HSV
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Remove low saturation (gray/dull) and low brightness (shadows)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    valid_mask = (sat > 40) & (val > 40)

    # Broad camouflage-green range: hue 30–90 degrees
    hue = hsv[:, :, 0]
    green_range = (hue >= 30) & (hue <= 90)

    # Combine both masks
    combined = valid_mask & green_range

    # Calculate ratio
    greenish_pixels = np.count_nonzero(combined)
    total_pixels = torso.shape[0] * torso.shape[1]

    ratio = greenish_pixels / total_pixels

    # Decide based on threshold
    return ratio > 0.30   # >= 30% green = military


# ----------------------------
# Main code
# ----------------------------
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load image
img = cv2.imread('Billeder/IMG_20251104_094347.jpg')
resized = cv2.resize(img, None, fx=0.2, fy=0.2)

# Detect people
boxes, weights = hog.detectMultiScale(
    resized,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.02
)

# Process detections
for (x, y, w, h) in boxes:
    # Determine if military jacket
    is_military = detect_military_jacket(resized, (x, y, w, h))

    # Draw boxes
    if is_military:
        color = (0, 255, 0)  # green box
        label = "MILITARY"
    else:
        color = (0, 0, 255)  # red box
        label = "CIVILIAN"

    cv2.rectangle(resized, (x, y), (x + w, y + h), color, 2)
    cv2.putText(resized, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Display result
cv2.imshow('Military Detection', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
