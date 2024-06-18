import torch
import numpy as np

def detect_parking_spots(model_path, image_path):
    # Load the trained model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    
    # Perform inference
    results = model(image_path)

    # Extract bounding box coordinates
    bounding_boxes = results.xyxy[0].numpy()  # x1, y1, x2, y2, confidence, class

    # Filter by class (assuming class_id 0 is for parking spots)
    parking_spots = bounding_boxes[bounding_boxes[:, 5] == 0]

    # Extract coordinates
    coordinates = []
    for box in parking_spots:
        x1, y1, x2, y2, _, _ = box
        coordinates.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    return coordinates

if __name__ == "__main__":
    model_path = 'runs/train/exp/weights/best.pt'  # Replace with your trained model path
    image_path = 'path/to/your/parking_lot_image.jpg'  # Replace with your image path
    coordinates = detect_parking_spots(model_path, image_path)
    print("Detected parking spot coordinates:", coordinates)
