import cv2
import numpy as np

# --- 1. Indlæs billede ---
img = cv2.imread("Billeder/IMG_20251104_094347.jpg")
if img is None:
    raise ValueError("Kunne ikke indlæse billedet")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 2. Fjern støj ---
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# --- 3. Baggrunds-segmentering via adaptive threshold ---
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21, 10
)

# --- 4. Morfologi for at samle kroppen til én blob ---
kernel = np.ones((7, 7), np.uint8)
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

# --- 5. Connected Components ---
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean)

# --- 6. Filtrer blobs ---
min_area = 500
max_area = 50000

for label in range(1, num_labels):
    x, y, w, h, area = stats[label]

    if area < min_area or area > max_area:
        continue

    aspect = h / float(w)
    if aspect < 1.2:
        continue

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# --- 7. Skaler billeder til skærmvisning ---
def resize_for_display(image, width=800):
    h, w = image.shape[:2]
    scale = width / float(w)
    new_dim = (width, int(h * scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

img_display = resize_for_display(img)
clean_display = resize_for_display(clean)

# --- 8. Vis resultat ---
cv2.imshow("Detected person", img_display)
cv2.imshow("Mask", clean_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
