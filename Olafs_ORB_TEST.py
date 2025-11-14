import cv2
import numpy as np

# ----------------------------------------
# 1. Load reference image of the person
# ----------------------------------------
template_path = "Billeder\\reference_person.jpg"  # replace with your template image
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise FileNotFoundError(f"Template image not found: {template_path}")

# ----------------------------------------
# 2. Initialize ORB detector
# ----------------------------------------
orb = cv2.ORB_create(nfeatures=20000, scaleFactor=1.5, nlevels=8)

# Extract keypoints and descriptors from template
kp_template, des_template = orb.detectAndCompute(template, None)
if des_template is None or len(kp_template) == 0:
    raise ValueError("No ORB features found in template image.")

print(f"Template: {len(kp_template)} keypoints")

# ----------------------------------------
# 3. Initialize BFMatcher (Brute Force)
# ----------------------------------------
# Use NORM_HAMMING for ORB binary descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ----------------------------------------
# 4. Load video
# ----------------------------------------
cap = cv2.VideoCapture("ProjektVideoer\\Civil person.MP4")
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
    # 5. Detect ORB features in current frame
    # ----------------------------------------
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    output = frame.copy()
    if des_frame is not None and len(kp_frame) > 10:
        # Match descriptors (kNN with k=2 for ratio test)
        matches = bf.knnMatch(des_template, des_frame, k=2)

        # Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # tune threshold (0.7–0.8)
                    good_matches.append(m)

        print(f"Frame {frame_count}: {len(good_matches)} good matches")

        # ----------------------------------------
        # 6. Find bounding box around matched keypoints
        # ----------------------------------------
        if len(good_matches) > 15:  # require minimum matches (tune as needed)
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

            # Optional: draw matches for debugging
            # match_img = cv2.drawMatches(template, kp_template, gray, kp_frame,
            #                             good_matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow("Matches", match_img)

    cv2.imshow("ORB Person Matching", output)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()