import cv2

# ======================= Suckit constant =======================
RECV_LEN = 5
HOST = "192.168.23.249"
PORT = 2001

# ======================= Servo constant =======================
SERVO_NUM = 0
SERVO_ANGLE = 90
ANGLE = [90, 90, 90, 90]

# ======================= Motor constant =======================
LEFT_SPEED = 100
RIGHT_SPEED = 100

# ======================= XUltraSonic =======================
DISTANCE = 0

# ======================= GPIO PINS =======================
ENA = 13 # L298 enables A
ENB = 20 # L298 enable B

IN1 = 16 # Motor interface 1
IN2 = 19 # Motor interface 2
IN3 = 26 # Motor interface 3
IN4 = 21 # Motor interface 4

ECHO = 4 # Ultrasonic receiving pin
TRIG = 17 # Ultrasonic transmitting pin

IR_R = 18 # Infrared line patrol on the right side of the car
IR_L = 27 # Infrared line patrol on the left side of the car
IR_M = 22 # Infrared for obstacle avoidance in the middle of the car
IRF_R = 25 # The car follows the right infrared
IRF_L = 1 # The car follows the left red

# CV2 CONST
FPS_TEXT_COLOR = (54, 54, 255)
BB_TEXT_COLOR = (206, 89, 59)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 2
THICKNESS = 2
ORG_FPS = (7, 25)
