import numpy as np
import cv2
import time
import random


class ImageSave:
    def __init__(self):
        self.capture_time = 0

    def save(self, l, r):
        if (time.time() - self.capture_time > 2):
            name = str(random.randint(1, 100))
            cv2.imwrite("images/{}_l.jpg".format(name), l)
            cv2.imwrite("images/{}_r.jpg".format(name), r)
            self.capture_time = time.time()