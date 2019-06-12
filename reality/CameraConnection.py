import cv2


class CameraConnection:
    def __init__(self, resolution, location=-1):
        self._res = resolution
        self._loc = location

    def __enter__(self):
        print "getting cam"
        self.cam = cv2.VideoCapture(self._loc)
        print self.cam.isOpened()
        print "setting resolution" + str(self._res)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._res[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._res[1])
        print "resolution set"

        return self

    def get_image(self):
        print self.cam.get(3)
        print self.cam.get(4)
        print self.cam.get(5)
        print "getting image"
        ret, img = self.cam.read()
        print ret
        cv2.imshow('frame', img)
        return img

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
        cv2.VideoCapture(self._loc).release()
