import cv2
from ultralytics import YOLO
import numpy as np

# CONST
FPS_TEXT_COLOR = (54, 54, 255)
BB_TEXT_COLOR = (206, 89, 59)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 2
THICKNESS = 2
ORG_FPS = (7, 25)

path_model = "../model/front_camera_v1.pt"
model = YOLO(path_model)

for i in range(7):
    path = f'../frames/frame{i}.jpg'
    frame = cv2.imread(path)
    results = model(frame, verbose=False, max_det=1, classes=[4])

    # cv2.imshow("Frame", frame)

    for res in results:
        boxes = res.boxes
        names = res.names
        for box in boxes:
            if len(box.xyxy) < 1:
                continue

            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int values
            dy2 = (y2 - y1) // 2

            mid_dot1, mid_dot2 = (x1, y1+dy2), (x2, y2-dy2)
            # print(mid_dot1, mid_dot2)
            frame = cv2.circle(frame, mid_dot1, 10, (0, 0, 255), thickness=-1)
            frame = cv2.circle(frame, mid_dot2, 10, (0, 0, 255), thickness=-1)

            zc = 10
            fx, fy = 900, 905

            xc, yc = mid_dot1
            x_left_cube, y_left_cube = zc * xc / fx, zc * yc / fy

            xc, yc = mid_dot2
            x_right_cube, y_right_cube = zc * xc / fx, zc * yc / fy

            cube_x, cube_y = abs(x_right_cube - x_left_cube), abs(y_left_cube - y_right_cube)

            print(cube_x)

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = int((box.conf[0] * 100)) / 100

            # object details
            name = names[int(box.cls[0])]

            text_label = f"class: {name}, conf: {confidence: .2f}%"
            # print(text_label, int(box.cls[0]))
            org_label = [x1, y1-5]
            cv2.putText(frame, text_label, org_label, FONT, FONT_SCALE, BB_TEXT_COLOR, THICKNESS)

        cv2.imshow('Webcam', frame)

    cv2.waitKey(0)

# Release the capture and writer objects
cv2.destroyAllWindows()
