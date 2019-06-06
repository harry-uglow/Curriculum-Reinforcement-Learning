import cv2


class CameraConnection:
    def __init__(self, resolution, location=0) -> None:
        self.cam = cv2.VideoCapture(location)
        self._location = location
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def __enter__(self):
        return self

    def get_image(self):
        ret, img = self.cam.read()
        return img

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
        cv2.VideoCapture(self._location).release()
