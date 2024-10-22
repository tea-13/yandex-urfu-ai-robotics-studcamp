import cv2
import numpy as np


def get_mask(_image, low_color=(0, 0, 0), high_color=(0, 0, 0), kernel_size=(10, 10)):
    binary = cv2.inRange(_image, low_color, high_color)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    _mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker, iterations=1)

    return _mask


def get_mask_hsv(_image, _low_color=(0, 0, 0), _high_color=(0, 0, 0), _kernel_size=(10, 10)):
    image_hsv = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
    mask_hsv = get_mask(image_hsv, low_color=_low_color, high_color=_high_color, kernel_size=_kernel_size)
    # image_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_HSV2BGR)

    return mask_hsv


def get_contours(_image):
    # Нахождение контуров
    _contours, _ = cv2.findContours(_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка контуров по площади
    _contours = sorted(_contours, key=cv2.contourArea, reverse=True)

    return _contours


def get_cube_bb(_frame):
    mask = get_mask_hsv(_frame, (0, 100, 100), (255, 255, 255))
    contours = get_contours(mask)

    if len(contours) > 0:
        _x1, _y1, _w1, _h1 = cv2.boundingRect(contours[0])

        return _x1, _y1, _x1 + _w1, _y1 + _h1

    else:
        return None


for i in range(23):
    path = f'../frames/frame{i}.jpg'
    frame = cv2.imread(path)

    bb = get_cube_bb(frame)
    if bb is not None:
        x1, y1, x2, y2 = bb
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    cv2.waitKey(0)
