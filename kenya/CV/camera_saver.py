import cv2
import numpy as np

url = 'http://192.168.198.249:8080/?action=stream'
# Подключение камеры
cap = cv2.VideoCapture(url)

i = 0
def on_key_press(frame):
    global i
    path = f'../frames/frame{i}.jpg'
    cv2.imwrite(path, frame)
    print(f"Кадр сохранен как {path}")
    i += 1


# Основной цикл программы
while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Нажатие 'q' для выхода
        break

    elif key == ord('s'):  # Нажатие 's' для сохранения кадра
        on_key_press(frame)

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
