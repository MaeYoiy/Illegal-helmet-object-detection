import os
import cv2
from ultralytics import YOLO

#model_path = 'yolov8l_100e.pt'
model_path = 'D:/Graduation_Project/Illegal-helmet-object-detection/runs/detect/train10/weights/best.pt'
#model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

#Load model
model = YOLO(model_path) # Load a custom model

threshold = 0.5

# class_name_dict = {0: 'alpaca'}
class_name_dict = {
    0: 'helmet',
    1: 'illegal_helmet',
    2: 'no_helmet',
}

cap = cv2.VideoCapture(0)  # Use default camere
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
    cv2.imshow('Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
