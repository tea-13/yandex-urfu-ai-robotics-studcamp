import socket
import time
from time import monotonic


class Suckit:
    def __init__(self, _host: str, _port: int) -> None:
        self.s = None
        print(f"Установка {_host}:{_port}")
        self.host = _host
        self.port = _port

        # date filtering
        self.i = 0
        self.dist_filtered = 0
        self.distances = [0, 0, 0]

    def __enter__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Соединение с {self.host}:{self.port}")
        # Устанавливаем соединение
        self.s.connect((self.host, self.port))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.s is not None:
            print('Закрытие сокета')
            self.s.close()

    @staticmethod
    def middle_of_3(a: int, b: int, c: int) -> int:
        """
        Функция вычисляющая медиану 3-х чисел
        :param a: Значение с сонара
        :param b: Значение с сонара
        :param c: Значение с сонара
        :return: Медиана 3-х чисел
        """
        if a <= b and a <= c:
            middle = b if b <= c else c
        elif b <= a and b <= c:
            middle = a if a <= c else c
        else:
            middle = a if a <= b else b
        return middle

    def send_data(self, command, latency):
        print(f"Отправка команды: {command}")
        # Отправляем команду
        self.s.sendall(command)
        # Добавляем небольшой задержку между отправками команд
        time.sleep(latency)

    def get_data(self):
        data = self.s.recv(1024)

        if not data:
            return None

        return data

    def filtering_msg(self, data):
        data = list(data)

        distance = 200
        if data[0] == 255:
            distance = data[3]
        elif data[0] == 1:
            infra = data[-3:]
            return 1, infra

        self.distances[self.i % 3] = distance
        dist = self.middle_of_3(self.distances[0], self.distances[1], self.distances[2])
        delta = abs(self.dist_filtered - dist)

        if delta > 1:
            k = 0.7
        else:
            k = 0.1
        self.dist_filtered = dist * k + self.dist_filtered * (1 - k)

        self.i += 1

        return 2, self.dist_filtered