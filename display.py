import cv2
import numpy as np


class Display:
    def __init__(self):
        return

    def display(self, color, ball_coords, disp, map, r_mask, b_mask):
        cv2.imshow("Disparity Map", disp/np.max(disp))
        for coord in ball_coords:
            cv2.circle(color, (coord[0], coord[1]), 5, (0, 0, 255), -1)
        cv2.imshow("Color Img", color)
        map = map.astype(float)
        map = cv2.resize(map, (map.shape[1] * 4, map.shape[0] * 4), interpolation = cv2.INTER_AREA)
        cv2.imshow("Map", map)
        print(np.max(b_mask))
        cv2.imshow("r_mask", r_mask.astype(float))
        cv2.imshow("b_mask", b_mask.astype(float))