from ultralytics import YOLO
import numpy as np
import cv2

mtx = np.array([[773.26292765, 0, 570.13791164], [0, 778.62717956, 424.74947619], [0, 0, 1]])
dist = np.array([[-0.38884924, 0.20372825, -0.00198133, 0.00477834, -0.06288507]])

ORIG_SIZE = (1280, 720)


def defisheye(_img):
    # resize image
    _img = cv2.resize(_img, ORIG_SIZE)

    # ================================ Уборка искажения ============================================
    # =============================================================================================
    # Изменение высоты
    original_width, original_height = ORIG_SIZE
    new_height = 760
    height_to_add = new_height - original_height
    top_padding = height_to_add // 2
    bottom_padding = height_to_add - top_padding

    new_size = (new_height, original_width, 3)

    result = np.zeros(new_size, dtype=np.uint8)
    result[top_padding:top_padding + original_height, 0:original_width] = _img
    # =============================================================================================

    # =============================================================================================
    h, w = result.shape[:2]

    # убираем искажения
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(result, mtx, dist, None, newcameramtx)
    # =============================================================================================

    # =============================================================================================
    binary = cv2.inRange(dst, (0, 0, 0), (0, 0, 0))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker, iterations=1)

    # Нахождение контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка контуров по площади
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    x1, y1, w1, h1 = cv2.boundingRect(contours[0])
    x2, y2, w2, h2 = cv2.boundingRect(contours[1])

    # cv2.rectangle(dst, (x1, y1+h1), (x2+w2, y2), (0, 255, 0), 2)

    # =============================================================================================
    dh, dw = 50, 40
    dst = dst[y1 + h1 - dh:y2 + dh, x1 - dw:x2 + w2 + dh]
    # =============================================================================================
    new_h, new_w = 480, 640
    h, w, _ = dst.shape

    d = int((w - (new_w / new_h) * h)) // 2  # magic num сужает ширину
    dst = dst[:, d:-d + 1]
    h, w, _ = dst.shape
    dst = cv2.resize(dst, (640, 480))
    return dst
    # =============================================================================================


def get_obb_robot(x0, y0, xf, yf, _dst):
    """
    Получает на вход координаты баунда робота (верхнюю левую и нижнюю правую), баундбокс и цвет робота (зеленый = 0/красный = 1)
    Выдает угол в градусах
    """
    low_color = (0, 0, 0)
    high_color = (255, 150, 150)
    kernel_size = (3, 3)

    _crop_robot = _dst[y0:yf, x0:xf]

    ret_crop_robot = _crop_robot

    _crop_robot = cv2.cvtColor(_crop_robot, cv2.COLOR_BGR2HSV)

    # Всякае поеба с фильрами
    binary = cv2.inRange(_crop_robot, low_color, high_color)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilation = cv2.dilate(binary, kernel_size, iterations=2)
    mask = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, ker, iterations=1)

    # Нахождение контуров по маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Сортировка контуров по площади
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    _box = cv2.boxPoints(rect)

    _box = np.intp(_box)

    return ret_crop_robot, _box


def get_robot_angle(_img, _box):
    lower_white = np.array([0, 0, 200])  # Значение V должно быть высоким
    upper_white = np.array([180, 30, 255])  # С низкой насыщенностью для

    hsv = cv2.cvtColor(crop_robot, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    coords = np.column_stack(np.where(mask > 100))
    if len(coords) > 0:
        Dcent = list(map(int, np.mean(coords, axis=0)))
        light_dot = [Dcent[1], Dcent[0]]
        robot_dot = cv2.circle(crop_robot, light_dot, 5, (255, 0, 255), -1)

        dot_dist = lambda l: (l[0] - light_dot[0]) ** 2 + (l[1] - light_dot[1]) ** 2

        sorted_box = sorted(box, key=dot_dist)

        delta_x = sorted_box[2][0] - sorted_box[0][0]
        delta_y = sorted_box[2][1] - sorted_box[0][1]
        eps = 1e-6

        angle = np.rad2deg(np.atan2(delta_y, delta_x + eps))
        if angle < 0:
            angle += 360

        angle = 360 - angle

        return angle

    raise ValueError("Еблан угор не считается")


BB_TEXT_COLOR = (206, 89, 59)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 2
THICKNESS = 2
ORG_FPS = (7, 25)

if __name__ == '__main__':
    model = YOLO("../model/best_up_camera_v2.pt")

    cap = cv2.VideoCapture("rtsp://Admin:rtf123@192.168.2.251/251:554/1/1")

    while True:
        ret, img = cap.read()

        if not ret:
            break

        # =============================defisheye===================
        dst = defisheye(img)

        # =============================== model ===================
        results = model.track(dst, persist=True, verbose=False)

        # =========================bboxes and map====================
        for result in results:
            boxes = result.boxes
            # print(result.names)
            tracking_label = result.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, tracking_label):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy.astype(int)
                label = result.names[int(box.cls.item())]
                # print(label, track_id)
                # cv2.rectangle(dst, (x1, y1), (x2, y2), (255, 0, 0), 2)

                label_text = f"{label}, {track_id}"

                label_position = (x1, y1 - 10)  # Position the label slightly above the bounding box

                cv2.rectangle(dst, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(dst, label_text, label_position, FONT, FONT_SCALE, BB_TEXT_COLOR)

                if label == "robot":
                    crop_robot, box = get_obb_robot(x1, y1, x2, y2, dst)
                    try:
                        angle = get_robot_angle(crop_robot, box)
                        print(f"{track_id = } {angle = }")
                    except Exception:
                        pass

                    """# cont_img = cv2.drawContours(dst, [box], 0, (0, 255, 255), 2)
                    cont_img = cv2.line(dst, sorted_box[0], sorted_box[2], (0, 255, 255), 1)

                    color = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), ]  # B, R, G, LB
                    for p, c in zip(sorted_box, color):
                        cont_img = cv2.circle(cont_img, p, 5, c, -1)"""

                    """cv2.imshow('cont_img', cont_img)
                    cv2.imshow('robot_dot', robot_dot)"""

            cv2.imshow('Frame', dst)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
