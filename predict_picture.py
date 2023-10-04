from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('D:/Graduation_Project/Illegal-helmet-object-detection/runs/detect/train/weights/best.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('./Images/IMG4.jpeg', save=True)

# cv2.waitKey(0)
