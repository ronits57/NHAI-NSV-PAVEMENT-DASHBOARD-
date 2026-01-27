from ultralytics import YOLO
import os

# Load trained model ONCE
MODEL_PATH = r"C:\Users\Ronit\OneDrive\Desktop\road_ai\runs\detect\train5\weights\best.pt"
model = YOLO(MODEL_PATH)

def detect_road_damage(image_path):
    results = model(image_path, conf=0.25)

    potholes = 0
    cracks = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])

            if cls_id == 0:      # pothole
                potholes += 1
            elif cls_id == 1:    # crack
                cracks += 1

    total = potholes + cracks

    # Simple scoring
    health_score = max(0, 100 - (potholes * 5 + cracks * 3))

    if health_score > 80:
        condition = "Good"
        action = "No action required"
    elif health_score > 60:
        condition = "Fair"
        action = "Monitor and minor repairs"
    else:
        condition = "Poor"
        action = "Immediate maintenance required"

    return {
        "potholes": potholes,
        "cracks": cracks,
        "health_score": health_score,
        "condition": condition,
        "action": action
    }
