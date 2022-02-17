import numpy as np
import time

class Ball:
    def __init__(self, BALL_RADIUS):
        self.pos = [0,0,BALL_RADIUS]
        self.vel = [0,0,0]
        self.state = 0
        self.aerial = 0
        self.color = 0
        self.fresh = 0
        self.last_tracked = time.time()
    