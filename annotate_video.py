#!/usr/bin/env python3
"""
Simple video annotation tool for creating COCO JSON annotations.
Annotate frames by drawing bounding boxes around people.

Usage:
    python annotate_video.py --video <video_path> --output <json_output>
    
Controls:
    - Click and drag to draw bounding box
    - 'n' = next frame
    - 'p' = previous frame
    - 's' = skip 10 frames
    - 'd' = delete last box on current frame
    - 'c' = clear all boxes on current frame
    - 'q' = quit and save
    - SPACE = play/pause video
"""

import cv2
import json
import argparse
from pathlib import Path
import numpy as np


class VideoAnnotator:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_frame_id = 0
        self.frame_annotations = {}  # frame_id -> list of boxes
        self.drawing = False
        self.start_x, self.start_y = 0, 0
        self.current_box = None
        
        print(f"Video: {self.width}x{self.height}, {self.total_frames} frames, {self.fps:.1f} fps")
        print("Controls: Click+drag to draw box | n=next | p=prev | s=skip10 | d=delete | c=clear | q=quit")
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = (self.start_x, self.start_y, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = min(self.start_x, x), min(self.start_y, y)
                x2, y2 = max(self.start_x, x), max(self.start_y, y)
                
                # Store as [x, y, width, height] (COCO format)
                w = x2 - x1
                h = y2 - y1
                if w > 5 and h > 5:  # Ignore tiny boxes
                    if self.current_frame_id not in self.frame_annotations:
                        self.frame_annotations[self.current_frame_id] = []
                    self.frame_annotations[self.current_frame_id].append([x1, y1, w, h])
                    print(f"  Added box on frame {self.current_frame_id}: [{x1}, {y1}, {w}, {h}]")
                self.current_box = None
    
    def draw_frame(self, frame):
        """Draw boxes on frame."""
        display = frame.copy()
        
        # Draw existing boxes
        if self.current_frame_id in self.frame_annotations:
            for box in self.frame_annotations[self.current_frame_id]:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw current box being drawn
        if self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return display
    
    def goto_frame(self, frame_id):
        """Jump to specific frame."""
        frame_id = max(0, min(frame_id, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.current_frame_id = frame_id
    
    def run(self):
        """Main annotation loop."""
        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate", self.mouse_callback)
        
        playing = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.goto_frame(self.total_frames - 1)
                ret, frame = self.cap.read()
            
            display = self.draw_frame(frame)
            
            # Info text
            num_boxes = len(self.frame_annotations.get(self.current_frame_id, []))
            info = f"Frame: {self.current_frame_id+1}/{self.total_frames} | Boxes: {num_boxes} | {'PLAYING' if playing else 'PAUSED'}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Annotate", display)
            
            key = cv2.waitKey(30 if playing else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):  # Next frame
                self.goto_frame(self.current_frame_id + 1)
                playing = False
            elif key == ord('p'):  # Previous frame
                self.goto_frame(self.current_frame_id - 1)
                playing = False
            elif key == ord('s'):  # Skip 10 frames
                self.goto_frame(self.current_frame_id + 10)
                playing = False
            elif key == ord('d'):  # Delete last box
                if self.current_frame_id in self.frame_annotations and len(self.frame_annotations[self.current_frame_id]) > 0:
                    box = self.frame_annotations[self.current_frame_id].pop()
                    print(f"  Deleted box: {box}")
                    if len(self.frame_annotations[self.current_frame_id]) == 0:
                        del self.frame_annotations[self.current_frame_id]
            elif key == ord('c'):  # Clear frame
                if self.current_frame_id in self.frame_annotations:
                    del self.frame_annotations[self.current_frame_id]
                    print(f"  Cleared frame {self.current_frame_id}")
            elif key == ord(' '):  # Play/pause
                playing = not playing
            elif playing:
                self.goto_frame(self.current_frame_id + 1)
                if self.current_frame_id >= self.total_frames - 1:
                    playing = False
        
        cv2.destroyAllWindows()
        self.save_coco()
    
    def save_coco(self):
        """Export annotations as COCO JSON."""
        images = []
        annotations = []
        annotation_id = 1
        
        for frame_id in sorted(self.frame_annotations.keys()):
            image_id = frame_id
            images.append({
                "id": image_id,
                "file_name": f"frame_{frame_id:06d}.png",
                "width": self.width,
                "height": self.height,
            })
            
            for box in self.frame_annotations[frame_id]:
                x, y, w, h = box
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Person
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                annotation_id += 1
        
        coco_data = {
            "info": {
                "description": "Video annotation",
                "version": "1.0",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "human",
                }
            ],
        }
        
        with open(self.output_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\n✓ Saved {len(annotations)} annotations to: {self.output_path}")
        print(f"  Images: {len(images)}, Annotations: {len(annotations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate video frames with bounding boxes")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output COCO JSON path")
    args = parser.parse_args()
    
    annotator = VideoAnnotator(args.video, args.output)
    annotator.run()
