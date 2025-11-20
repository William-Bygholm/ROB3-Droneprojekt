import cv2
import numpy as np

# ----------------------------------------
# 1. Load reference image of the person
# ----------------------------------------
template_path = "Billeder\reference_person.jpg"  # replace with your template image
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(f"Template image not found: {template_path}")

# ----------------------------------------
# 2. Initialize SIFT detector
# ----------------------------------------
sift = cv2.SIFT_create()

# Extract keypoints and descriptors from template
kp_template, des_template = sift.detectAndCompute(template, None)
if des_template is None or len(kp_template) == 0:
    raise ValueError("No SIFT features found in template image.")

print(f"Template: {len(kp_template)} keypoints")

# ----------------------------------------
# 3. Initialize matcher (FLANN is faster)
# ----------------------------------------
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# ----------------------------------------
# 4. Load video
# ----------------------------------------
cap = cv2.VideoCapture("ProjektVideoer\Civil person.MP4")
resize_scale = 0.5  # adjust for speed vs accuracy

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ----------------------------------------
    # 5. Detect SIFT features in current frame
    # ----------------------------------------
    kp_frame, des_frame = sift.detectAndCompute(gray, None)

    output = frame.copy()
    if des_frame is not None and len(kp_frame) > 10:
        # Match descriptors
        matches = matcher.knnMatch(des_template, des_frame, k=2)

        # Lowe's ratio test to filter good matches
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        print(f"Frame {frame_count}: {len(good_matches)} good matches")

        # ----------------------------------------
        # 6. Find bounding box around matched keypoints
        # ----------------------------------------
        if len(good_matches) > 10:  # require minimum matches
            # Get coordinates of matched keypoints in frame
            pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute bounding box (min-max of matched points)
            x_coords = pts[:, 0, 0]
            y_coords = pts[:, 0, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())

            # Add margin
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(frame.shape[1], x_max + margin)
            y_max = min(frame.shape[0], y_max + margin)

            # Draw box and label
            cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output, f"PERSON ({len(good_matches)} matches)",
                        (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Optional: draw matches (for debugging)
            # match_img = cv2.drawMatches(template, kp_template, gray, kp_frame,
            #                             good_matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow("Matches", match_img)

    cv2.imshow("SIFT Person Matching", output)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()