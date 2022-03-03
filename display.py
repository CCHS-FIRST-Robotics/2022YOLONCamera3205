import cv2
import numpy as np


def nothing(*arg):
    pass


class Display:
    def __init__(self):
        return

    def display(self, color, ball_coords):
        for coord in ball_coords:
            if coord[2] == "R":
                cv2.circle(color, (coord[0], coord[1]), round(coord[3]), (0, 0, 255), 2)
            else:
                cv2.circle(color, (coord[0], coord[1]), round(coord[3]), (255, 0, 0), 2)
        cv2.imshow("Color Img", color)

        return
