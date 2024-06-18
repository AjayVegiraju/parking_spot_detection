import os
import torch
from pathlib import Path

def train_yolov5(data_yaml, img_size=640, batch_size=16, epochs=50, weights='yolov5s.pt'):
    # Train the YOLOv5 model
    os.system(f"python yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights {weights}")

if __name__ == "__main__":
    data_yaml = 'data.yaml'
    train_yolov5(data_yaml)
