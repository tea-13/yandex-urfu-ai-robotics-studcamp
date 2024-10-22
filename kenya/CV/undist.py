import numpy as np
import cv2

from tracker import Tracker, FONT, FONT_SCALE, BB_TEXT_COLOR, THICKNESS


class ImageProcessing:
    mtx = np.array([
        [773.26292765, 0, 570.13791164],
        [0, 778.62717956, 424.74947619],
        [0, 0, 1]
    ])
    dist = np.array([[-0.38884924, 0.20372825, -0.00198133, 0.00477834, -0.06288507]])

    ORIG_SIZE = (1280, 720)

    def __init__(self):
        self.chanel_num = 3

    def reshape(self, _image, new_height=760):
        image = cv2.resize(_image, self.ORIG_SIZE)

        # Изменение высоты
        original_width, original_height = self.ORIG_SIZE
        height_to_add = new_height - original_height
        top_padding = height_to_add // 2

        new_size = (new_height, original_width, self.chanel_num)

        # Создание нового изображения с отступами сверху и снизу
        result = np.zeros(new_size, dtype=np.uint8)
        result[top_padding:top_padding + original_height, 0:original_width] = image

        return result

    def clean_distortion(self, _image):
        h, w = _image.shape[:2]

        # Создаем оптимальную матрицу
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        # убираем искажения
        # undistort
        __undistortion = cv2.undistort(_image, self.mtx, self.dist, None, newcameramtx)

        return __undistortion

    def get_mask(self, _image, low_color=(0, 0, 0), high_color=(0, 0, 0), kernel_size=(10, 10)):
        binary = cv2.inRange(_image, low_color, high_color)
        ker = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker, iterations=1)

        return mask

    def get_mask_hsv(self, _image, _low_color=(0, 0, 0), _high_color=(0, 0, 0), _kernel_size=(10, 10)):
        image_hsv = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
        mask_hsv = self.get_mask(image_hsv, low_color=_low_color, high_color=_high_color, kernel_size=_kernel_size)
        image_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_HSV2BGR)

        return image_bgr

    def get_contours(self, _image):
        # Нахождение контуров
        contours, _ = cv2.findContours(_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Сортировка контуров по площади
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        return contours

    def proportional_resize(self, _image, new_size=(640, 480)):
        h, w, _ = _image.shape

        _new_w, _new_h = new_size

        d = int((w - (_new_w / _new_h) * h)) // 2  # magic num сужает ширину
        _image = _image[:, d:1 - d, :]
        h, w, _ = _image.shape

        _frame = cv2.resize(_image, new_size)

        return _frame

    def crop_image(self, _image, _x1, _x2, _y1, _y2, _dh, _dw):
        return _image[_y1 - _dh:_y2 + _dh, _x1 - _dw:_x2 + _dh]


# url = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"
url = "rtsp://Admin:rtf123@192.168.2.251/251:554/1/1"
path = r'../videos/1.avi'
cap = cv2.VideoCapture(url)
# traker = Tracker("../model/best_up_camera_v2.pt", verbose=False, stream=True, conf=0.7, iou=0.2, max_det=10)  # classes=[0]

while True:
    ret, img = cap.read()

    if not ret:
        break

    # =============================================================================================
    image_processing = ImageProcessing()
    reshape_img = image_processing.reshape(img)
    undistortion = image_processing.clean_distortion(reshape_img)
    mask = image_processing.get_mask(undistortion)
    contours = image_processing.get_contours(mask)

    # =============================================================================================
    x1, y1, w1, h1 = cv2.boundingRect(contours[0])
    x2, y2, w2, h2 = cv2.boundingRect(contours[1])

    # cv2.rectangle(dst, (x1, y1+h1), (x2+w2, y2), (0, 255, 0), 2)
    # =============================================================================================
    dh, dw = 40, 40
    dst = image_processing.crop_image(undistortion, x1, x2 + w2, y1 + h1, y2, dh, dw)
    # =============================================================================================
    new_h, new_w = 480, 640

    dst = image_processing.proportional_resize(dst)

    cv2.imshow("frame", dst)
    # cv2.imshow('img', dst)
    # =============================================================================================
    '''
    frame = dst

    results = traker.model(frame, verbose=False, max_det=15, conf=0,
                           iou=0)  # , conf=0.4, iou=0.7, max_det=1) # persist=True

    for res in results:
        boxes = res.boxes
        for box in boxes:
            if len(box.xyxy) < 1:
                continue

            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int values

            # confidence
            confidence = int((box.conf[0] * 100)) / 100

            if confidence < 0.7:
                continue

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # object details
            text_label = f"class: {box.cls[0]}, conf: {confidence: .2f}%"
            org_label = [x1, y1 - 5]
            cv2.putText(frame, text_label, org_label, FONT, 1, BB_TEXT_COLOR, THICKNESS)

        cv2.imshow('Webcam', frame)

    # ======================================= Цвета ===============================================
    # =============================================================================================
    dst_hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

    green_min = (20, 100, 100)
    green_max = (70, 255, 255)

    binary = cv2.inRange(dst_hsv, green_min, green_max)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker, iterations=1)

    # Нахождение контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка контуров по площади
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(dst, (x, y), (x + w, y + h), (255, 255, 0), 2)'''

    # cv2.imshow('img', dst)
    # =============================================================================================

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
