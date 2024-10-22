# URL для подключения к IP-камере. Это может быть RTSP или другой протокол потокового видео
import cv2
url = 'http://192.168.23.249:8080/?action=stream'
# url = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"
# url = "rtsp://Admin:rtf123@192.168.2.251/251:554/1/1" # Создаем объект VideoCapture для захвата видео с IP-камеры
cam = cv2.VideoCapture(url)


def compare_histograms(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    # Используем метод Бхаттачария
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return similarity


prev_frame = None
while True:
    ret, frame = cam.read()

    print(ret)

    (h, w) = frame.shape[:2]

    # Desired width
    new_width = 800

    # Calculate the aspect ratio
    aspect_ratio = h / w
    new_height = int(new_width * aspect_ratio)

    frame = cv2.resize(frame, (new_width, new_height))

    if prev_frame is not None:
        sim = compare_histograms(frame, prev_frame)
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