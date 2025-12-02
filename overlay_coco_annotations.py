import cv2
import json
import os

# Configuration
VIDEO_PATH = r"ProjektVideoer/3 mili 2 onde 1 god.MP4"  # Update with your video filename
COCO_JSON = "3mili 2 onde 1 god.json"
OUTPUT_VIDEO = "Output/annotated_video.mp4"

# Colors for different classes (BGR format)
CLASS_COLORS = {
    "Military good": (255, 0, 0),      # Blue
    "Military bad": (0, 0, 255),       # Red
    "Good HVT": (255, 255, 0),         # Cyan
    "Bad HVT": (0, 128, 255),          # Orange
    "Civilian": (128, 128, 128),       # Gray
    "Unknown person": (0, 255, 255)    # Yellow
}

def load_coco_annotations(json_path):
    """Load COCO JSON and organize annotations by frame"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image_id to annotations
    frame_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in frame_annotations:
            frame_annotations[image_id] = []
        frame_annotations[image_id].append(ann)
    
    # Create mapping from image_id to frame number (if available)
    image_info = {img['id']: img for img in data['images']}
    
    return frame_annotations, image_info

def get_bbox_class(annotation):
    """Extract the active class from annotation attributes"""
    attrs = annotation.get('attributes', {})
    for class_name, is_active in attrs.items():
        if is_active and class_name in CLASS_COLORS:
            return class_name
    return "Unknown person"

def draw_annotation(frame, annotation):
    """Draw a single bounding box with label on the frame"""
    bbox = annotation['bbox']  # COCO format: [x, y, width, height]
    x, y, w, h = [int(v) for v in bbox]
    
    # Get class and color
    class_name = get_bbox_class(annotation)
    color = CLASS_COLORS.get(class_name, (255, 255, 255))
    track_id = annotation.get('attributes', {}).get('track_id', '?')
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Draw label with background
    label = f"{class_name} ID:{track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y), color, -1)
    cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
    
    return frame

def main():
    # Load annotations
    print("Loading COCO annotations...")
    frame_annotations, image_info = load_coco_annotations(COCO_JSON)
    print(f"Loaded {len(frame_annotations)} frames with annotations")
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # Process video
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # COCO image_id might start from a different number - find matching annotations
        # Try to match by frame index or image_id
        matching_annotations = []
        
        # Try different possible image_id values
        for possible_id in [frame_idx, frame_idx + 1, 3720 + frame_idx]:  # Adjust offset as needed
            if possible_id in frame_annotations:
                matching_annotations = frame_annotations[possible_id]
                break
        
        # Draw all annotations for this frame
        for ann in matching_annotations:
            frame = draw_annotation(frame, ann)
        
        # Write frame
        out.write(frame)
        
        # Display progress
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}/{total_frames} ({frame_idx*100//total_frames}%)")
        
        # Optional: Display frame (press 'q' to quit early)
        display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('Annotated Video', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nDone! Annotated video saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
