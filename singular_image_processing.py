import numpy as np
import cv2
from pathlib import Path


def turn_video_to_images(video_path, max_dim=None):
    """Reads a video file and returns a list of frames as images.
    
    Args:
        video_path (str or Path): Path to the video file.
        max_dim (int, optional): Maximum dimension (width or height) for resizing frames.
                                 If None, original size is kept.
    
    Returns:
        List of frames (images) read from the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # optional resize to limit memory/display size
        if max_dim is not None:
            h, w = frame.shape[:2]
            scale = min(1.0, max_dim / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        frames.append(frame.copy())  # store a copy to be safe

    cap.release()
    return frames

def turn_images_to_gray_scale(images):
    gray_scale_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    return gray_scale_images





def view_images_one_by_one(gray_images):
    if not gray_images:
        print("No images to display.")
        return

    idx = 0
    winname = "Gray Viewer"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    while True:
        gray = gray_images[idx]
        # convert to BGR so we can draw coloured text
        disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp, f"{idx+1}/{len(gray_images)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(winname, disp)
        key = cv2.waitKey(0) & 0xFF  # wait until a key is pressed

        if key == 27 or key == ord('q'):   # ESC or q -> quit
            break
        elif key == ord('n'):              # n -> next
            idx = min(idx + 1, len(gray_images) - 1)
        elif key == ord('b'):              # b -> back
            idx = max(idx - 1, 0)
        # other keys ignored; still stay on current frame

    cv2.destroyWindow(winname)


def edge_detection_on_pictures(gray_images):
    edge_images = []
    for gray in gray_images:
        edges = cv2.Canny(gray, 100, 200)
        edge_images.append(edges)
    return edge_images

def detect_people_by_contours(edge_images, orig_images=None, min_area=500, max_area_ratio=0.6):
    """
    Find bounding boxes from binary/edge images by contour detection.
    If orig_images provided, return boxes relative to those images (useful for drawing).
    """
    all_boxes = []
    for i, edge in enumerate(edge_images):
        # ensure binary (Canny outputs 0/255 already but enforce)
        _, bw = cv2.threshold(edge, 50, 255, cv2.THRESH_BINARY)
        # close gaps and remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

        contours_info = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        h_img, w_img = (edge.shape[0], edge.shape[1])
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if area > (w_img * h_img * max_area_ratio):
                continue
            # basic aspect ratio filter for people-like shapes (tweak as needed)
            ar = h / (w + 1e-6)
            if ar < 0.6:  # too wide -> likely not a person
                continue
            boxes.append((x, y, w, h))
        all_boxes.append(boxes)
    return all_boxes

def view_detections(images, boxes, window_name="Detections"):
    if not images:
        print("No images to display.")
        return
    idx = 0
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        img = images[idx].copy()
        for (x, y, w, h) in boxes[idx]:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{idx+1}/{len(images)}  Boxes:{len(boxes[idx])}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n'):
            idx = min(idx + 1, len(images) - 1)
        elif key == ord('b'):
            idx = max(idx - 1, 0)
    cv2.destroyWindow(window_name)

images = turn_video_to_images("Billeder/Elias walking - Copy.mp4", max_dim=800)
gray_images = turn_images_to_gray_scale(images)
edge_images = edge_detection_on_pictures(gray_images)
people_boxes = detect_people_by_contours(edge_images, orig_images=images)
view_detections(images, people_boxes, window_name="People Detections")

