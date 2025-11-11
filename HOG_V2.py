import cv2

# -------------------------
# Parameters
# -------------------------
video_path = "C:/Users/alexa/Downloads/Civil person.MP4"   # Path to your drone video
resize_scale = 0.5                   # Resize frames for faster processing

# Load Haar cascade for full body or upper body detection
# You can try haarcascade_fullbody.xml or haarcascade_upperbody.xml
cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
human_cascade = cv2.CascadeClassifier(cascade_path)

if human_cascade.empty():
    print("Error: cannot load cascade xml:", cascade_path)
    exit()

# -------------------------
# Open video
# -------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: cannot open video:", video_path)
    exit()

# -------------------------
# Processing loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans
    humans = human_cascade.detectMultiScale(
        gray,
        scaleFactor=1.03,
        minNeighbors=4,
        minSize=(30, 60),  # Minimum size of person in pixels (tune based on drone altitude)
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw bounding boxes
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, max(0, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Detected Humans", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()
