import numpy as np

class BallSim:
    def __init__(self, TOWER_RAD, BALL_RADIUS):
        self.center_r = TOWER_RAD
        self.ball_r = BALL_RADIUS
        self.dims = [16.46, 8.23]
        self.g = -9.81
        self.dt = 0.01
        self.r_fric = 0.05
        self.drag = 0.000136180501711 / 0.270
        self.bounce_prop = 0.7/0.9144
    def vertical_comp(self, height, v_vel):
        down_vel = v_vel + self.g * self.dt
        naive_pred_height = height + v_vel * self.dt + 0.5 * self.g * self.dt**2
        if naive_pred_height < self.ball_r:
            down_vel = down_vel * -1 * self.bounce_prop
            naive_pred_height = -1 * naive_pred_height
        return naive_pred_height, down_vel
    def horiz_comp(self, pos, vels):
        #on floor check
        xy = np.array(pos[0:2])
        xy_vi = np.array(vels[0:2])
        xy_v_mag = (vels[0]**2 + vels[1]**2)**0.5
        xy_v = xy_vi
        if pos[2] < self.ball_r*2 and abs(vels[2]) < 0.2:
            xy_v = xy_vi - self.dt * self.r_fric * xy_vi/xy_v_mag
            if xy_v_mag < self.dt * self.r_fric:
                xy_v = np.array([0,0])
        xy = xy + (xy_v + xy_vi)*0.5

        # left x wall
        if (xy[0] < -1 * self.dims[0]/2):
            xy_v[0] = xy_v[0] * -1 * self.bounce_prop
            xy[0] = xy[0] - (xy[0] + self.dims[0]/2) * 2
        # right x wall
        if (xy[0] > self.dims[0]/2):
            xy_v[0] = xy_v[0] * -1 * self.bounce_prop
            xy[0] = xy[0] - (xy[0] - self.dims[0]/2) * 2
        # Top

    def predict_bounce(self):