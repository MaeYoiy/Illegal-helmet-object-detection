# Import Flask and other required libraries
from flask import Flask, Response

# Required to run the YOLOv8 model
import cv2

# Import video_detection function from YOLO_Video
from YOLO_Video import video_detection

app = Flask(__name__)

app.config['SECRET_KEY'] = 'detecthelmetmink'

# Define a global variable to store the video capture object
cap = None

# Initialize the video capture object
# def initialize_camera(path_x):
#     global cap
#     # You can specify the path to your video file here or use a webcam by providing 0
#     cap = cv2.VideoCapture(path_x)

# Generate_frames function now takes frames as input and yields them as output
def generate_frames(path_x):
    # initialize_camera(path_x)
    
    while True:
        # success, frame = cap.read()

        # if not success:
        #     break

        # Perform object detection on the frame
        # yolo_output = video_detection(frame)
        yolo_output = video_detection(path_x)

        for detection_frame in yolo_output:
            _, buffer = cv2.imencode('.jpg', detection_frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        # If you want to add additional frames, you can yield them here
        # For example, you can add a static image after each video frame
        static_image = cv2.imread('static_image.jpg')
        _, buffer = cv2.imencode('.jpg', static_image)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(path_x='../Videos/test_vid2.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
