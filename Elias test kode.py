from ultralytics import YOLO
import cv2

# === Parametre ===
VIDEO_PATH = "ProjektVideoer/2 mili en idiot der ligger ned.MP4"  # ret til din video
CONF_THRESH = 0.4
SCALE = 0.5  # 0.5 = 50% størrelse, kan justeres

# === YOLO model ===
model = YOLO("yolov8n.pt")

# === Video ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Kunne ikke åbne videoen")

# Opret et vindue du kan resize
cv2.namedWindow("YOLOv8 Human Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Human Detection", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Kør YOLO detektion
    results = model.predict(frame, conf=CONF_THRESH, verbose=False)

    # Tegn kun mennesker (class 0)
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Human {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Skaler preview ned
    h, w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (int(w * SCALE), int(h * SCALE)))

    # Vis billedet
    cv2.imshow("YOLOv8 Human Detection", frame_resized)

    # Tryk ESC for at lukke
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
