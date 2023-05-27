import cv2
import mediapipe as mp
from face_tracking import *
import pyautogui
from time import time

class Tracking:
    def __init__(self, mp_face_detection, mp_drawing, face_detection):
        self.mp_face_detection = mp_face_detection
        self.mp_drawing = mp_drawing
        self.face_detection = face_detection

    def process(self, image):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return (image, results)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
 
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.1) as face_detection:
    tracking = Tracking(mp_face_detection, mp_drawing, face_detection)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image, results = tracking.process(image)
        if results.detections:
            start_time = time()
            ispassed = True
            while ispassed:
                success, image = cap.read()
                if not success:
                    break
                image, results = tracking.process(image)
                cv2.imshow("Autolock Face Detection", image)
                if cv2.waitKey(1) == ord("w"):
                    break
                if not results.detections:
                    end_time = time()
                    gap = end_time - start_time
                else:
                    ispassed = False
                    continue
                if gap >= 2:
                    pyautogui.hotkey("ctrl", "command", "q")
                else:
                    ispassed = True
        cv2.imshow("Autolock Face Detection", image)
        if cv2.waitKey(1) == ord("w"):
            break
      
cap.release()
cv2.destroyAllWindows()
