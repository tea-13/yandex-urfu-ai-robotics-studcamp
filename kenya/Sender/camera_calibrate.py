import cv2
from numpy import np

ip_camera_url_left = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"
cap = cv2.VideoCapture(ip_camera_url_left)#video_capture = cv2.VideoCapture(ip_camera_url_right)

def calibrate_camera(square_size):
    # Создаем объект камеры
    camera_matrix = np.zeros((3, 3))
    dist_coeffs = np.zeros((5,))

    # Создаем объект для хранения точек
    objpoints = []
    imgpoints = []

    # Параметры для захвата изображения
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Цикл для захвата нескольких изображений
    for i in range(20):  # Захватим 20 изображений
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Находим угол квадрата в изображении
        ret, corners = cv2.findChessboardCorners(gray, (square_size, square_size))

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Рисуем квадрат на изображении для визуализации
            cv2.drawChessboardCorners(frame, (square_size, square_size), corners, ret)

    # Выполняем калибровку
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],
                                                                        camera_matrix, dist_coeffs, criteria=criteria)

    return camera_matrix, dist_coeffs