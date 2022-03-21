import numpy as np
import time
from BallTrack.BallDynamics import BallSim


class Ball:
    def __init__(self, BALL_RADIUS):
        self.BALL_RADIUS = BALL_RADIUS
        self.reset()
        self.cooldown_time = 0

    def reset(self):
        self.pos = [0, 0, self.BALL_RADIUS]
        self.prev_pos = self.pos
        self.vel = [0, 0, 0]
        self.state = 0
        self.aerial = 0
        self.color = 0
        self.fresh = 0 #0 fresh, 1 stable
        self.last_tracked = time.time()
        self.sim = BallSim(self.BALL_RADIUS)
        self.cooldown_time = time.time()

    def init(self, pos, color, state = 0):
        if color == "R":
            self.color = 0
        else:
            self.color = 1
        self.pos = pos
        self.vel = [0, 0, 0]
        self.prev_pos = pos
        if self.pos[2] < self.BALL_RADIUS * 3:
            self.aerial = 0
        else:
            self.aeriel = 1
        self.state = state
        self.fresh = 0

    def predict(self, dt):
        if self.fresh == 0:
            return
        self.sim.dt = dt
        self.prev_pos = self.pos
        self.pos, self.vel = self.sim.simulate(self.pos, self.vel)
        if self.pos[2] < self.BALL_RADIUS * 3:
            self.aerial = 0
        else:
            self.aerial = 1

    def update(self, emp_pos):
        dt = time.time() - self.last_tracked
        emp_vel = (np.array(emp_pos) - np.array(self.prev_pos)) / dt
        emp_vel = [emp_vel[0], emp_vel[1], emp_vel[2]]
        pf = 0.5
        vf = 0.5
        self.pos = [self.pos[0] * pf + emp_pos[0] * (1 - pf), self.pos[1] * pf + emp_pos[1] * (1 - pf),
                    self.pos[2] * pf + emp_pos[2] * (1 - pf)]
        self.vel = [self.vel[0] * vf + emp_vel[0] * (1 - vf), self.vel[1] * vf + emp_vel[1] * (1 - vf),
                    self.vel[2] * vf + emp_vel[2] * (1 - vf)]
        self.last_tracked = time.time()
        self.fresh = 1
        if self.pos[2] < self.BALL_RADIUS * 3:
            self.aerial = 0
        else:
            self.aerial = 1

    def getDist(self, x2):
        return ((x2[0] - self.pos[0])**2 + (x2[1] - self.pos[1])**2 + (x2[2] - self.pos[2])**2)**0.5

    def getSpd(self):
        return (self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)**0.5
