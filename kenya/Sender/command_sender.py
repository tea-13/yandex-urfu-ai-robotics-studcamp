from __future__ import with_statement

import numpy as np
from typing import Callable, Tuple
import socket
import time

host = "192.168.23.249"
port = 2001
latency: float = .00


class ConcretUnit:
    """
    Класс генерации команд управления
    Все номера передаются в десятичном формате

    Из таблицы
    sensor_number - Вид устройства
    device_number - Управляющий байт
    additional_number - Байт данных
    """

    def __init__(self, _sensor_number: int):
        self.sensor_number = ConcretUnit.short_hex(_sensor_number)

    def get_command(self, device_number: int, additional_number: int):
        """
        Функция для генерации команд

        :param device_number: (int) Управляющий байт
        :param additional_number: (int) Байт данных

        :return: (bytes) Байт-послед. конечная команда
        """
        device_number = ConcretUnit.short_hex(device_number)
        additional_number = ConcretUnit.short_hex(additional_number)

        # Формирование команды ввиде hex-последовательности
        hex_sequence = 'ff'
        hex_sequence += ' '
        hex_sequence += self.sensor_number
        hex_sequence += ' '
        hex_sequence += device_number
        hex_sequence += ' '
        hex_sequence += additional_number
        hex_sequence += ' '
        hex_sequence += 'ff'

        # Формирование команды ввиде байт-последовательности
        bytes_sequence = bytes.fromhex(hex_sequence)

        return bytes_sequence

    @staticmethod
    def short_hex(num: int) -> str:
        """
        Функция переводит в hex и убирает префикс "0x"

        :param num: (int) Число представленое для обработки

        :return: (str) Конечный результат в формате XX
        """
        if not isinstance(num, int):
            raise TypeError("Нужен int")

        try:
            hex_num = hex(num)[2:].zfill(2)
        except ValueError as e:
            print()
            return ''

        return hex_num


class UnitFactory:
    """
    Класс Фабрика для генерации конкретных функций управления
    """

    def __init__(self):
        self.motor_direction = None
        self.motor_speed = None
        self.servo = None
        self.car_lights = None

        self.build()  # Создание всех необходимых устройств

        self.device_message_function = {
            'motor_direction': self.motor_direction.get_command,
            'servo': self.servo.get_command,
            'motor_speed': self.motor_speed.get_command,
            'car_lights': self.car_lights.get_command,
        }

    def build(self):
        """
        Создание всех элементов фабрики
        :return: None
        """
        self.build_motor_direction()
        self.build_motor_speed()
        self.build_servo_bus()
        self.build_car_lights()

    def build_motor_direction(self):
        sensor_number = 0
        self.motor_direction = ConcretUnit(sensor_number)

    def build_motor_speed(self):
        sensor_number = 2
        self.motor_speed = ConcretUnit(sensor_number)

    def build_servo_bus(self):
        sensor_number = 1
        self.servo = ConcretUnit(sensor_number)

    def build_car_lights(self):
        sensor_number = 6
        self.car_lights = ConcretUnit(sensor_number)

    def get_instance(self, device_name: str) -> Callable:
        return self.device_message_function.get(device_name, None)


def main_led(func_msg):
    latency = 1
    i = 0

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Соединение с {host}:{port}")

    # Устанавливаем соединение
    s.connect((host, port))

    while True:
        command = func_msg(i % 5 + 1, i % 8 + 1)

        try:
            print(f"Отправка команды: {command}")

            # Отправляем команду
            s.sendall(command)

            # Добавляем небольшой задержку между отправками команд
            time.sleep(latency)

        except socket.error as e:
            print(f"Ошибка сокета: {e}")
            break

        except KeyboardInterrupt:
            print("Выход по клавиатуре.")
            break

        i += 1

    if s is not None:
        print('Закрытие сокета')
        s.close()


def main_servo(func_msg):
    s = None
    i = 0

    while True:
        command = func_msg(2, i % 90)

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print(f"Соединение с {host}:{port}")

            # Устанавливаем соединение
            s.connect((host, port))
            print(f"Отправка команды: {command}")

            # Отправляем команду
            s.sendall(command)

            # Добавляем небольшой задержку между отправками команд
            time.sleep(latency)

        except socket.error as e:
            print(f"Ошибка сокета: {e}")
            break

        except KeyboardInterrupt:
            print("Выход по клавиатуре.")
            break

        i += 1

    if s is not None:
        print('Закрытие сокета')
        s.close()


def main_motor(func_msg1, func_msg2):
    latency = 0.1
    SPEED = 40
    s = None
    i = 0

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Соединение с {host}:{port}")

    # Устанавливаем соединение
    s.connect((host, port))

    while True:
        if i == 0:
            command = func_msg2(1, SPEED)
        elif i == 1:
            command = func_msg2(2, SPEED)
        else:
            command = func_msg1(1, 0)
            latency = 100

        try:
            # Отправляем команду
            s.sendall(command)

            # Добавляем небольшой задержку между отправками команд
            time.sleep(latency)

        except socket.error as e:
            print(f"Ошибка сокета: {e}")
            break

        except KeyboardInterrupt:
            print("Выход по клавиатуре.")
            break

        i += 1

    if s is not None:
        print('Закрытие сокета')
        s.close()


def one_comand_sender(func_msg, a, b):
    command = func_msg(a, b)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Соединение с {host}:{port}")

    # Устанавливаем соединение
    s.connect((host, port))
    print(f"Отправка команды: {command}")

    try:
        # Отправляем команду
        s.sendall(command)

        # Добавляем небольшой задержку между отправками команд
        time.sleep(latency)

    except socket.error as e:
        print(f"Ошибка сокета: {e}")

    except KeyboardInterrupt:
        print("Выход по клавиатуре.")

    if s is not None:
        print('Закрытие сокета')
        s.close()


def line_interpol_servo(func_msg, servonum, a, b, step):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Соединение с {host}:{port}")

    # Устанавливаем соединение
    s.connect((host, port))

    for t in np.arange(0.00, 1.00 + step, step):
        ang = int(b * t + a * (1 - t))
        print(ang)
        command = func_msg(servonum, ang)

        try:
            # Отправляем команду
            s.sendall(command)

            # Добавляем небольшой задержку между отправками команд
            time.sleep(latency)

        except socket.error as e:
            print(f"Ошибка сокета: {e}")
            break

        except KeyboardInterrupt:
            print("Выход по клавиатуре.")
            break

    if s is not None:
        print('Закрытие сокета')
        s.close()


class Sender:
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
        print(f"Соединение с {host}:{port}")
        # Устанавливаем соединение
        self.s.connect((host, port))
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
            return 0, 0

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


class IK:
    def __init__(self, _l1, _l2, _dx, _dy):
        self.l1 = _l1
        self.l2 = _l2
        self.dx = _dx
        self.dy = _dy

    def calculate(self, x: float, y: float) -> Tuple[int, int]:
        x += self.dx
        y += self.dy

        b = x ** 2 + y ** 2
        q1 = np.atan2(y, x)
        q2 = np.acos((self.l1 ** 2 - self.l2 ** 2 + b) / (2 * self.l1 * np.sqrt(b)))

        phi1 = np.rad2deg(q1 + q2)

        phi2 = np.acos((self.l1 ** 2 + self.l2 ** 2 - b) / (2 * self.l1 * self.l2))
        phi2 = np.rad2deg(phi2)

        return round(phi1), round(phi2)


if __name__ == "__main__":
    factory = UnitFactory()

    motor_direction_msg = factory.get_instance('motor_direction')
    motor_speed_msg = factory.get_instance('motor_speed')
    servo_msg = factory.get_instance('servo')
    led_mgs = factory.get_instance('car_lights')

    # main_led(led_mgs)
    # main_servo(servo_msg)
    # main_motor(motor_direction_msg, motor_speed_msg)
    one_comand_sender(led_mgs, 1, 8)
    # one_comand_sender(servo_msg, 4, 92)
    #line_interpol_servo(servo_msg, 3, 0, 180, .01)
    # one_comand_sender(servo_msg, 3, 100)  # наклон
    # one_comand_sender(servo_msg, 7, 0)  # поворот
    #exit(0)
    #one_comand_sender(motor_speed_msg, 1, 20)
    #one_comand_sender(motor_speed_msg, 2, 25)

    #latency: float = 5
    #one_comand_sender(motor_direction_msg, 1, 0)
    exit(0)
    with Sender(host, port) as s:
        servo1 = lambda ang: servo_msg(1, ang)
        servo2 = lambda ang: servo_msg(2, ang)
        servo3 = lambda ang: servo_msg(3, ang)
        servo4 = lambda ang: servo_msg(4, ang)
        servo7 = lambda ang: servo_msg(7, ang)
        servo8 = lambda ang: servo_msg(8, ang)

        ik = IK(0.1, 0.12,  0.05, -0.11)

        s.send_data(servo4(0), 0.1)


