import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'D:/Graduation_Project/Illegal-helmet-object-detection/')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

#Check if the video opened successfully
if not cap.isOpened():
    print("Error openning video.")
    exit()

ret, frame = cap.read()

#Check if the first frame was read successfully
if not ret:
    print("Error reading the first frame.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.videoWrier_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'best.pt')