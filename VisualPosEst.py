import numpy as np



BALL_RADIUS = 0.12

X_FOV = 53
Y_FOV = 39.75
X_DIST = 0.15
RPP = 2 * 3.1415 * 53 / (360 * 640)

def ball2coords(cam_shape, lballs, rballs, index):
    given_ball = lballs[index]
    color = given_ball[2]
    min_y_disp = 50000
    min_index = -1
    for c in range(len(rballs)):
        ball = rballs[c]
        if ball[2] == color:
            if ball[0] > given_ball[0]:
                if abs(ball[1] - given_ball[1]) < min_y_disp:
                    min_y_disp = abs(ball[1] - given_ball[1])
                    min_index = c
    if min_y_disp < 10:
        xl_t, yl_t = pixel2Theta(cam_shape, given_ball[0], given_ball[1])
        xr_t, yr_t = pixel2Theta(cam_shape, rballs[min_index][0], rballs[min_index][1])
        #
        left_triangle_angle = 90 + xl_t
        right_triangle_angle = 90 - xr_t
        #sin(rta)/dist = sin(xr_t - xl_t) / X_DIST
        dist = X_DIST * np.sin(right_triangle_angle) / (np.sin(xr_t - xl_t) + 0.001)
        x_disp = np.sin(x_theta) * dist
        z_disp = np.sin(y_theta) * dist
        y_disp = np.cos(x_theta) * dist
        return [x_disp, y_disp, z_disp]
    else:
        return pixel2LPos(cam_shape, given_ball[0], given_ball[1], given_ball[3])

def pixel2Theta(cam_shape, x_pixel, y_pixel):
    return RPP * (cam_shape[1]*0.5 - x_pixel), RPP * (cam_shape[0]*0.5 - y_pixel)

def pixel2LPos(cam_shape, x_pixel, y_pixel, radius):
    #get x angle, y angle. Distance is num of pixels of radius gives theta
    ball_angle = RPP * radius
    #tan(theta) = radius / dist
    distance = BALL_RADIUS / np.tan(ball_angle)
    x_theta = RPP * (x_pixel - cam_shape[1]*0.5)
    y_theta = RPP * (cam_shape[0]*0.5 - y_pixel)
    x_disp = np.sin(x_theta) * distance
    z_disp = np.sin(y_theta) * distance
    y_disp = np.cos(x_theta) * distance
    return [x_disp, y_disp, z_disp]