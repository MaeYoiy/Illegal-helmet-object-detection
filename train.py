from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='D:\Graduation_Project\Illegal-helmet-object-detection\datasets\Illegal-Helmet-Detection-8\data.yaml', epochs=100, imgsz=640)


