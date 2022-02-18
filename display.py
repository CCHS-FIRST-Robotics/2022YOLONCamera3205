import cv2
import numpy as np


def nothing(*arg):
    pass


class Display:
    def __init__(self):
        cv2.namedWindow('trackbars_r')

        cv2.createTrackbar('lowHue', 'trackbars_r', 0, 255, nothing)
        cv2.createTrackbar('lowSat', 'trackbars_r', 0, 255, nothing)
        cv2.createTrackbar('lowVal', 'trackbars_r', 0, 255, nothing)
        # Higher range colour sliders.
        cv2.createTrackbar('highHue', 'trackbars_r', 0, 255, nothing)
        cv2.createTrackbar('highSat', 'trackbars_r', 0, 255, nothing)
        cv2.createTrackbar('highVal', 'trackbars_r', 0, 255, nothing)

        cv2.namedWindow('trackbars_b')

        cv2.createTrackbar('lowHue', 'trackbars_b', 0, 255, nothing)
        cv2.createTrackbar('lowSat', 'trackbars_b', 0, 255, nothing)
        cv2.createTrackbar('lowVal', 'trackbars_b', 0, 255, nothing)
        # Higher range colour sliders.
        cv2.createTrackbar('highHue', 'trackbars_b', 0, 255, nothing)
        cv2.createTrackbar('highSat', 'trackbars_b', 0, 255, nothing)
        cv2.createTrackbar('highVal', 'trackbars_b', 0, 255, nothing)

        return

    def display(self, color, ball_coords, disp, map, r_mask, b_mask):
        cv2.imshow("Disparity Map", disp / np.max(disp))
        for coord in ball_coords:
            if coord[2] == "R":
                cv2.circle(color, (coord[0], coord[1]), coord[3], (0, 0, 255), 2)
            else:
                cv2.circle(color, (coord[0], coord[1]), coord[3], (255, 0, 0), 2)
        cv2.imshow("Color Img", color)
        map = map.astype(float)
        map = cv2.resize(map, (map.shape[1] * 4, map.shape[0] * 4), interpolation=cv2.INTER_AREA)
        cv2.imshow("Map", map)
        print(np.max(b_mask))
        cv2.imshow("r_mask", r_mask.astype(float))
        cv2.imshow("b_mask", b_mask.astype(float))

        range_mat = [0, 0, 0, 0, 0, 0]
        range_mat[0] = cv2.getTrackbarPos('lowHue', 'trackbars_r')
        range_mat[1] = cv2.getTrackbarPos('lowSat', 'trackbars_r')
        range_mat[2] = cv2.getTrackbarPos('lowVal', 'trackbars_r')
        range_mat[3] = cv2.getTrackbarPos('highHue', 'trackbars_r')
        range_mat[4] = cv2.getTrackbarPos('highSat', 'trackbars_r')
        range_mat[5] = cv2.getTrackbarPos('highVal', 'trackbars_r')

        bange_mat = [0, 0, 0, 0, 0, 0]
        bange_mat[0] = cv2.getTrackbarPos('lowHue', 'trackbars_b')
        bange_mat[1] = cv2.getTrackbarPos('lowSat', 'trackbars_b')
        bange_mat[2] = cv2.getTrackbarPos('lowVal', 'trackbars_b')
        bange_mat[3] = cv2.getTrackbarPos('highHue', 'trackbars_b')
        bange_mat[4] = cv2.getTrackbarPos('highSat', 'trackbars_b')
        bange_mat[5] = cv2.getTrackbarPos('highVal', 'trackbars_b')

        return range_mat, bange_mat
