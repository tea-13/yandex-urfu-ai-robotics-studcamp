from __future__ import with_statement

import numpy as np
from typing import Callable, Tuple

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