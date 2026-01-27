from ultralytics import YOLO
import cv2
import os

# Load trained model ONCE
MODEL_PATH = r"C:\Users\Ronit\OneDrive\Desktop\road_ai\runs\detect\train5\weights\best.pt"
model = YOLO(MODEL_PATH)

def detect_road_damage(image_path):
    img = cv2.imread(image_path)

    results = model.predict(
        source=img,
        conf=0.25,
        save=False
    )

    potholes = 0
    cracks = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # class names come from training
            if cls == 0:   # pothole
                potholes += 1
            elif cls == 1: # crack
                cracks += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_dir = "ai_outputs"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

    health_score = max(0, 100 - (potholes * 10 + cracks * 5))

    if health_score >= 80:
        condition = "Good"
        action = "Routine monitoring"
    elif health_score >= 60:
        condition = "Fair"
        action = "Preventive maintenance"
    else:
        condition = "Poor"
        action = "Immediate repair required"

    return {
        "potholes": potholes,
        "cracks": cracks,
        "health_score": health_score,
        "condition": condition,
        "action": action,
        "output_image": os.path.basename(image_path)
    }
