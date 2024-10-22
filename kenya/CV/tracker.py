import time
import math
from typing import List, Tuple

from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import cv2

# CONST
FPS_TEXT_COLOR = (54, 54, 255)
BB_TEXT_COLOR = (206, 89, 59)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 2
THICKNESS = 2
ORG_FPS = (7, 25)


class Tracker:
    """
    Tracker class is a wrapper for YOLO detector.
    The first person who gets into the frame is detected and
    YOLO continues to track him.
    """

    def __init__(self, path_model: str, **kwargs) -> None:
        self.model = YOLO(path_model)

        self.kwargs = kwargs  # Параметры для YOLO модели

        self.results = None
        self.boxes = None
        self.coordinates = []
        self.id = []

    def update(self, _frame: np.array) -> List:
        """
        The method processes the image

        :param _frame: The image that needs to be processed.
        """
        self.results = self.model.predict(source=_frame, **self.kwargs)

        results_list = list(self.results)

        return results_list

    def get_bb(self, _frame: np.array) -> List:
        """
        The method return the bounding box

        :param _frame: The image that needs to be processed.
        """
        results = self.update(_frame)

        return [result.boxes for result in results]


if __name__ == '__main__':
    traker = Tracker("../model/front_camera_v1.pt", verbose=False, stream=True, conf=0.5, iou=0.5)  # persist=True, classes=[0]

    url = 'http://192.168.2.85:8080/?action=stream'
    path = r'../videos/front160.mp4'
    vid = cv2.VideoCapture(path)

    # Resolution 640x480
    #vid.set(3, 640)
    #vid.set(4, 480)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        success, frame = vid.read()
        boxes = []

        #print(f"{frame_width}x{frame_height} :{success}")

        if success:


            # Image processing
            results = traker.model(frame, verbose=False)

            # FPS counter
            new_frame_time = time.time()
            fps_text = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {fps_text : .0f}"
            cv2.putText(frame, fps_text, ORG_FPS, FONT, FONT_SCALE, FPS_TEXT_COLOR)  # draw FPS counter

            for res in results:
                boxes = res.boxes
                names = res.names
                for box in boxes:
                    if len(box.xyxy) < 1:
                        continue

                    # bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int values

                    # put box in cam
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = int((box.conf[0] * 100)) / 100

                    # object details
                    text_label = f"class: {names[int(box.cls[0])]}, conf: {confidence: .2f}%"
                    print(text_label)
                    org_label = [x1, y1-5]
                    cv2.putText(frame, text_label, org_label, FONT, FONT_SCALE, BB_TEXT_COLOR, THICKNESS)

                cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
