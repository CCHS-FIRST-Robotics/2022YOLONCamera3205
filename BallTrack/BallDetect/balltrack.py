from __future__ import division
import cv2
from cv2 import Param_UNSIGNED_INT
import numpy as np
import time

from distortion import *

def nothing(*arg):
        pass

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Initial HSV GUI slider values to load on program start.
# icol = (36, 202, 59, 71, 255, 255)    # Green
# icol = (6, 156, 158, 30, 255, 255)  # Yellow
# icol = (89, 30, 60, 125, 220, 220)  # Blue
icol = (0, 62, 80, 20, 220, 255)   # Red
# icol = (0, 140, 90, 40, 255, 130) # Orange
# icol = (104, 117, 222, 121, 255, 255)   # test
# icol = (0, 0, 0, 255, 255, 255)   # New start

cv2.namedWindow('colorTest')
cv2.namedWindow('trackbars')

test = True
if not test:
    def print(*args):
        return

# Lower range colour sliders.
cv2.createTrackbar('lowHue', 'trackbars', icol[0], 255, nothing)
cv2.createTrackbar('lowSat', 'trackbars', icol[1], 255, nothing)
cv2.createTrackbar('lowVal', 'trackbars', icol[2], 255, nothing)
# Higher range colour sliders.
cv2.createTrackbar('highHue', 'trackbars', icol[3], 255, nothing)
cv2.createTrackbar('highSat', 'trackbars', icol[4], 255, nothing)
cv2.createTrackbar('highVal', 'trackbars', icol[5], 255, nothing)

# Initialize webcam. Webcam 0 or webcam 1 or ...
vidCapture = cv2.VideoCapture(0)
vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH)
vidCapture.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)

# initialize prev lists
prev = [] # prev contour list
prev_centers = [] # prev circle centers
prev_radii = [] # prev circle radii
past_ten = [] # past 10 (centers, radii, contours)
velocity = 0 # estimated velocity

# for distortion
images = []
get_images = False

pause = False # for pausing script/camera (helps w/ using sliders)

# used for choosing when to calculate velocity vectors
vector_num = 0

# hsv values of the sliders 
# (if you find a setting you like, close out of program 
#  using esc and it'll print the hsv vals)
saved_vals = None

fps = 15 # save fps vals
prev_fps = [] # saves past 10 fps vals
while True:
    # try to calculate velocity once every second 
    vector_num += 1
    vector_num = vector_num % int(sum(prev_fps) / len(prev_fps))

    timeCheck = time.time()
    # Get HSV values from the GUI sliders.
    lowHue = cv2.getTrackbarPos('lowHue', 'trackbars')
    lowSat = cv2.getTrackbarPos('lowSat', 'trackbars')
    lowVal = cv2.getTrackbarPos('lowVal', 'trackbars')
    highHue = cv2.getTrackbarPos('highHue', 'trackbars')
    highSat = cv2.getTrackbarPos('highSat', 'trackbars')
    highVal = cv2.getTrackbarPos('highVal', 'trackbars')
    hsv_vals = lowHue, lowSat, lowVal, highHue, highSat, highVal
    # print(f"({lowHue}, {lowSat}, {lowVal})")
    # print(f"({highHue}, {highSat}, {highVal})")

    if pause:
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        elif k == 99: # c
            print('boop2')
            get_images = True
        elif k == 112: # p
            print('boop3')
            pause = not pause
        print('fps - ', 1/(time.time() - timeCheck))
        continue

    # Get webcam frame
    _, frame = vidCapture.read()

    # Add frames to queue of images
    if get_images:
        print('boop1')
        if len(images) >= 15:
            images = []

        images.append(frame)
        if len(images) == 15:
            settings = camera_settings(images)
            get_images = False

    # Undistort the image
    if len(images) == 15:
        print('all gud')
        frame = undistort(frame, settings)

    # Show the original image.
    cv2.imshow('frame', frame)

    ### GREYSCALE ###
    # converting frame into grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    cv2.imshow('greyscale', threshold)
    ### GREYSCALE ###


    ### MASK ###
    # Convert the frame to HSV colour model.
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV values to define a colour range we want to create a mask from.
    colorLow = np.array([lowHue,lowSat,lowVal])
    colorHigh = np.array([highHue,highSat,highVal])
    mask = cv2.inRange(frameHSV, colorLow, colorHigh)

    # Show the first mask   
    cv2.imshow('mask-plain', mask)
    ### MASK ###

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # NOT USING RN - SLOW (not sure why i even tried it lol) #
    def get_avg_color(frame, x, y, r):
        height, width, depth = frame.shape
        total = []
        for i in range(height):
            for j in range(width):
                if (j - x) ** 2 + (i - y) ** 2 < r ** 2:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    color = hsv[i, j][0]
                    total.append(color)

        return sum(total) / len(total)
    #     #     #     #     #

    # find contours which match a circle
    good_contour_shape = []
    for i, contour in enumerate(contours):
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # color test thing #
        x,y,w,h = cv2.boundingRect(contour) # offsets - with this you get 'mask'
        color = np.array(cv2.mean(mask[y:y+h,x:x+w])).astype(np.uint8)[0]
        #         #        #

        # Check conditions
        if radius > 8 and cv2.contourArea(contour) > .5 * (np.pi * radius ** 2) and (color / 255) > .5:
            good_contour_shape.append((np.pi * (radius ** 2), cv2.contourArea(contour), contour))
            cv2.drawContours(frame, contour, -1, (255,0,0), 3)

    print('num_circles', len(good_contour_shape))
    
    for _, __,contour in good_contour_shape:
        # cv2.drawContours(frame, contour, -1, (0,255,0), 3)

        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        # print(x, type(x))

        # color = get_avg_color(frame, x, y, radius)
        # print("COLOR1:", color)

        circle = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        cv2.circle(circle, (int(x), int(y)), int(radius), 255, -1)
        color = cv2.mean(circle, mask=circle)

        # print("COLOR:", color)

        cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)

        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        
        # anything similar: dist from each center gets summed
        # now that I think about this i should probably just use avg (x, y) then calc distance
        similar = np.array([0.0, 0.0])
        for centers, radii, ___ in  past_ten:
            for center, oldr in zip(centers, radii):
                
                # check for similarity (not a great metric tbh)
                oldx, oldy = center
                if (oldx - 10 < x < oldx + 10) and (oldy - 10 < y < oldy + 10) and\
                    oldr - 5 < radius < oldr + 5:
                    similar += np.array([fps*(oldx - x), fps*(oldy - y)])

        # take the avg of the similar objects and calculate velocity
        if vector_num == 0:
            print("TEXT THING WORKING")
            velocity = (similar / len(similar)).round(2)

        cv2.putText(frame, str(velocity), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # save previous contours/shapes
    prev = good_contour_shape
    prev_centers = [cv2.minEnclosingCircle(contour)[0] for _, __,contour in good_contour_shape]
    prev_radii = [cv2.minEnclosingCircle(contour)[1] for _, __,contour in good_contour_shape]
    
    if len(past_ten) == 10:
        past_ten = past_ten[1:]
    past_ten.append((prev_centers, prev_radii, prev))


    # cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    
    #cv2.drawContours(frame, contours, 3, (0,255,0), 3)
    
    #cnt = contours[1]
    #cv2.drawContours(frame, [cnt], 0, (0,255,0), 3)

    # Show final output image
    cv2.imshow('colorTest', frame)

    # check for keypresses (this often is not instant, idk y opencv is so buggy sometimes)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: # esc
        print("SAVED VALS:", saved_vals)
        break
    elif k == 99: # c
        print('boop2')
        get_images = True
    elif k == 112: # p
        print('boop3')
        pause = True
    elif k == 115: # s
        print('saved!')
        saved_vals = hsv_vals

    # record fps
    fps = 1/(time.time() - timeCheck)
    if len(prev_fps) >= 10:
        prev_fps = prev_fps[1:]
    prev_fps.append(fps)

    print('fps - ', fps, '\t avg fps (past 10 frames) -- ', sum(prev_fps) / len(prev_fps))
    
cv2.destroyAllWindows()
vidCapture.release()