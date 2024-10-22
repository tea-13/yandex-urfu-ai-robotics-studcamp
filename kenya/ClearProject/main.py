import cv2

from Sender import Suckit
from Factory import UnitFactory

from Config import HOST, PORT

factory = UnitFactory()

motor_direction_msg = factory.get_instance('motor_direction')
motor_speed_msg = factory.get_instance('motor_speed')
servo_msg = factory.get_instance('servo')
led_mgs = factory.get_instance('car_lights')

def compare_histograms(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    # Используем метод Бхаттачария
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return similarity

url = 'http://192.168.23.249:8080/?action=stream'
# url = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"
# url = "rtsp://Admin:rtf123@192.168.2.251/251:554/1/1" # Создаем объект VideoCapture для захвата видео с IP-камеры
cam = cv2.VideoCapture(url)


prev_frame = None
with Suckit(HOST, PORT) as s:
    start = motor_direction_msg(1, 0)
    back = motor_direction_msg(2, 0)
    stop = motor_direction_msg(0, 0)

    while True:
        ret, frame = cam.read()

        s.send_data(start, 1)
        s.send_data(stop, 0.5)
        s.send_data(back, 1)
        s.send_data(stop, 0.5)

        if prev_frame is not None:
            sim = compare_histograms(frame, prev_frame)
            print(sim)
            cv2.imshow("Prev", prev_frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

        prev_frame = frame

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()
