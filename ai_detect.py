from ultralytics import YOLO
import cv2
import os

# LOAD YOUR TRAINED MODEL (IMPORTANT)
MODEL_PATH = r"C:\Users\Ronit\OneDrive\Desktop\road_ai\runs\detect\train4\weights\best.pt"
model = YOLO(MODEL_PATH)

def detect_road_damage(image_path):
    results = model(image_path, conf=0.2)

    pothole_count = 0
    output_image_path = None

    for r in results:
        pothole_count = len(r.boxes)

        # Save image with boxes
        img = r.plot()
        os.makedirs("ai_outputs", exist_ok=True)
        output_image_path = os.path.join("ai_outputs", "result.jpg")
        cv2.imwrite(output_image_path, img)

    # Simple road health logic
    health_score = max(0, 100 - pothole_count * 5)

    if pothole_count == 0:
        condition = "Good"
        action = "No action needed"
    elif pothole_count <= 3:
        condition = "Fair"
        action = "Monitor road condition"
    elif pothole_count <= 6:
        condition = "Poor"
        action = "Schedule maintenance"
    else:
        condition = "Very Poor"
        action = "Immediate repair required"

    return {
        "potholes": pothole_count,
        "health_score": health_score,
        "condition": condition,
        "action": action,
        "output_image": "result.jpg"
    }
