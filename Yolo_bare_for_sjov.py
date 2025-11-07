from ultralytics import YOLO
import cv2
import glob
import os

# --- SETTINGS ---
image_folder = "Billeder"   # mappen hvor billeder ligger
output_folder = "Output"    # YOLO-resultater

os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")  

# Find alle billeder i folder
extensions = ["*.jpg", "*.jpeg", "*.png"]
image_paths = []
for ext in extensions:
    image_paths.extend(glob.glob(os.path.join(image_folder, ext)))

print(f"Fandt {len(image_paths)} billeder")

for path in image_paths:
    img = cv2.imread(path)
    results = model(img)[0]

    # Tegn kun mennesker (class 0)
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Gem output-billedet
    filename = os.path.basename(path)
    out_path = os.path.join(output_folder, filename)
    cv2.imwrite(out_path, img)

    # Vis det
    cv2.imshow("YOLO Result", img)
    cv2.waitKey(300) 

cv2.destroyAllWindows()
