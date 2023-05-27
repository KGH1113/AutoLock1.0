import cv2

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

    def get_face_pos(self, image, results, draw):
        for detection in results.detections:
            face_pos = self.mp_face_detection.get_key_point(
                detection, self.mp_face_detection.FaceKeyPoint.NOSE_TIP
            )
            if draw:
                self.mp_drawing.draw_detection(image, detection)
            face_pos.x = 1 - face_pos.x
            face_pos.y = 1 - face_pos.y

        return face_pos