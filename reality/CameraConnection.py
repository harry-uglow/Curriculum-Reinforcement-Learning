import cv2
import numpy as np


class CameraConnection:
    def __init__(self, resolution, location=-1):
        self._res = resolution
        self._loc = location

    def __enter__(self):
        print "getting cam"
        self.cam = cv2.VideoCapture(self._loc)
#       print self.cam.isOpened()
        print "setting resolution" + str(self._res)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self._res[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self._res[1])
#       print "resolution set"

        return self

    def get_image(self):
        ret, img = self.cam.read()
        #img = np.array(cv2.cvtColor(cv2.imread('im_0.png'), cv2.COLOR_BGR2RGB))
        img = np.flipud(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return np.pad(img[:120, :120, :], ((4, 4), (4, 4), (0, 0)), 'edge')

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
        cv2.VideoCapture(self._loc).release()
