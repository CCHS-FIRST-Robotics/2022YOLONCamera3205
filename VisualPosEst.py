import numpy as np



BALL_RADIUS = 0.12

X_FOV = 53
Y_FOV = 39.75
RPP = 2 * 3.1415 * 53 / (360 * 640)

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