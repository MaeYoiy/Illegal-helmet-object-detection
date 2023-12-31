from ultralytics import YOLO
import cv2
import math
import os

# cap=cv2.VideoCapture('../Videos/test_vid2.mp4')

VIDEOS_DIR = os.path.join('.', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'D:/Graduation_Project/Illegal-helmet-object-detection/Videos/test7.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

# frame_width=int(cap.get(3))
# frame_height = int(cap.get(4))

ret, frame = cap.read()
H, W, _ = frame.shape

# out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, (frame_width, frame_height))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model=YOLO("D:/Graduation_Project/Illegal-helmet-object-detection/runs/detect/train10/weights/best.pt")
classNames = ["helmet", "illegal_helmet", "no_helmet"]


while True:
    success, img = cap.read()
    # Doing detections using YOLOv8 frame by frame
    #stream = True will use the generator and it is more efficient than normal
    results=model(img,stream=True)
    #Once we have the results we can check for individual bounding boxes and see how well it performs
    # Once we have have the results we will loop through them and we will have the bouning boxes for each of the result
    # we will loop through each of the bouning box
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            #print(x1, y1, x2, y2)
            x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
            #print(box.conf[0])
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            class_name=classNames[cls]
            label=f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            #print(t_size)
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            if class_name == "helmet":
                color=(0, 204, 255)
            elif class_name == "illegal_helmet":
                color = (222, 82, 175)
            elif class_name == "no_helmet":
                color = (0, 149, 255)
            else:
                color = (85,45,255)
            # if conf>0.5:
            cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
            cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
            # cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
            # cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
    out.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('1'):
        break
out.release()
