import cv2
from ultralytics import YOLO
import numpy as np

from Sender import Suckit
from Factory import UnitFactory
from IK import IK
from Config import HOST, PORT, THICKNESS, FONT, FONT_SCALE, BB_TEXT_COLOR

# Подключение камеры
url = 'http://192.168.198.249:8080/?action=stream'
cap = cv2.VideoCapture(url)

path_model = "../model/best_front_v2.pt"
model = YOLO(path_model)

# IK
ik = IK(0.1, 0.145, 0.06, -0.12)

# Comand sender
factory = UnitFactory()

motor_direction_msg = factory.get_instance('motor_direction')
motor_speed_msg = factory.get_instance('motor_speed')
servo_msg = factory.get_instance('servo')
led_mgs = factory.get_instance('car_lights')


def distance_to_camera(image, knownWidth=5, focalLength=900):
    """
    Принимает на вход изображение
    Выдает дистанцию до кубика


    Если будем делать шарик, то в параметрах knownWidth ставить 4.5
    """
    target_color, tolerance = (60, 0, 170), 255
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hue1 = np.array([0, 100, 100])
    upper_hue1 = np.array([90, 255, 255])
    lower_hue2 = np.array([140, 100, 100])
    upper_hue2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_hue1, upper_hue1)
    mask2 = cv2.inRange(hsv_image, lower_hue2, upper_hue2)
    hue_mask = cv2.bitwise_or(mask1, mask2)

    b, g, r = cv2.split(image)
    color_mask = (
            (np.abs(b - target_color[0]) <= tolerance) &
            (np.abs(g - target_color[1]) <= tolerance) &
            (np.abs(r - target_color[2]) <= tolerance)
    )

    combined_mask = cv2.bitwise_and(hue_mask, hue_mask, mask=hue_mask)
    combined_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=color_mask.astype(np.uint8) * 255)

    output_image = image.copy()

    output_image[combined_mask > 0] = [0, 255, 0]

    mask = combined_mask

    height, width = mask.shape
    max_height = 0
    start_y = -1
    end_y = -1

    for y in range(height):
        if np.any(mask[y, :] > 0):
            if start_y == -1:
                start_y = y
            end_y = y
        else:
            if start_y != -1:
                current_height = end_y - start_y + 1
                if current_height > max_height:
                    max_height = current_height
                start_y = -1
    if start_y != -1:
        current_height = end_y - start_y + 1
        if current_height > max_height:
            max_height = current_height

    if max_height == 0:
        return None
    else:
        return (knownWidth * focalLength) / max_height


def cube_linear_width(_mid_dot1, _mid_dot2, _zc):
    fx, fy = 900, 905

    xc, yc = _mid_dot1
    x_left_cube, y_left_cube = _zc * xc / fx, _zc * yc / fy

    xc, yc = _mid_dot2
    x_right_cube, y_right_cube = _zc * xc / fx, _zc * yc / fy

    _cube_x, _cube_y = abs(x_right_cube - x_left_cube), abs(y_left_cube - y_right_cube)

    return _cube_x


def ik_servo_rotate(dist):
    if dist <= 0.145:
        theta1, theta2 = ik.calculate(dist, 0.03)
        print(theta1, theta2)
        if 0 < theta1 < 180 and 0 < theta2 < 180:
            lt = .5
            c = servo1(theta1)
            s.send_data(c, lt)
            c = servo2(theta2)
            s.send_data(c, lt)
            c = servo3(90)
            s.send_data(c, lt)
            c = servo4(80)
            s.send_data(c, lt)
            c = servo1(60)
            s.send_data(c, lt)
            c = servo2(133)
            s.send_data(c, lt)
            c = servo4(80)
            s.send_data(c, lt)


# Основной цикл программы
with Suckit(HOST, PORT) as s:
    servo1 = lambda ang: servo_msg(1, ang)
    servo2 = lambda ang: servo_msg(2, ang)
    servo3 = lambda ang: servo_msg(3, ang)
    servo4 = lambda ang: servo_msg(4, ang)
    servo7 = lambda ang: servo_msg(7, ang)
    servo8 = lambda ang: servo_msg(8, ang)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # ============================== YOLO result ================================
        results = model(frame, verbose=False, max_det=1, classes=[4])

        zc = distance_to_camera(frame)
        # ===========================================================================

        # =============================== Results ===================================
        for res in results:
            boxes = res.boxes
            names = res.names
            for box in boxes:
                if len(box.xyxy) < 1:
                    continue

                # bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int values

                # midpoint bb
                dy2 = (y2 - y1) // 2
                mid_dot1, mid_dot2 = (x1, y1 + dy2), (x2, y2 - dy2)

                cube_x = cube_linear_width(mid_dot1, mid_dot2, zc)

                print(f"{zc= } {cube_x= }")

                # ======================================== YOLO labeling =======================================

                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

                # draw midpoint
                frame = cv2.circle(frame, mid_dot1, 2, (0, 0, 255), thickness=-1)
                frame = cv2.circle(frame, mid_dot2, 2, (0, 0, 255), thickness=-1)

                # confidence
                confidence = int((box.conf[0] * 100)) / 100

                # object details
                name = names[int(box.cls[0])]

                text_label = f"class: {name}, conf: {confidence: .2f}%"
                # print(text_label, int(box.cls[0]))
                org_label = [x1, y1 - 5]
                cv2.putText(frame, text_label, org_label, FONT, FONT_SCALE, BB_TEXT_COLOR, THICKNESS)
                # ========================================================================================

            cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == ord('q'):  # Нажатие 'q' для выхода
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
