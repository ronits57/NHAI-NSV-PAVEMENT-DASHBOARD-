import cv2
import os
from ai_detect_yolo import detect_road_damage

def analyze_video(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)

    total_potholes = 0
    
    frame_count = 0

    os.makedirs("video_frames", exist_ok=True)

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Take 1 frame every `frame_interval`
        if idx % frame_interval == 0:
            frame_file = f"video_frames/frame_{idx}.jpg"
            cv2.imwrite(frame_file, frame)

            result = detect_road_damage(frame_file)
            total_potholes += result["potholes"]
            
            frame_count += 1

        idx += 1

    cap.release()

    # Average over frames
    if frame_count > 0:
        avg_potholes = total_potholes // frame_count
        
    else:
        avg_potholes = 0
       

    return {
        "frames_analyzed": frame_count,
        "avg_potholes": avg_potholes,
    }
