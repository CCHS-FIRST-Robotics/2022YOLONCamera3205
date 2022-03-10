from networktables import NetworkTables


class CNetworkTable:
    def __init__(self):
        NetworkTables.initialize(server="10.32.5.2")
        self.table = NetworkTables.getTable('balls')
        self.state_table = NetworkTables.getTable('State')

    def setBallTable(self, ball_list):
        for c in range(len(ball_list)):
            pos_str = "pos_" + str(c)
            vel_str = "vel_" + str(c)
            g_str = "gstate_" + str(c)

            ball = ball_list[c]

            self.table.putNumberArray(pos_str, ball.pos)
            self.table.putNumberArray(vel_str, ball.vel)
            self.table.putNumberArray(g_str, [ball.state, ball.aerial, ball.color, ball.fresh])

            if (ball.state != 0 and ball.color == 0):
                print("Ball: {}".format(ball.pos))

    def getOdometry(self, odo):
        x = self.state_table.getNumber("x_pos", 0)
        y = self.state_table.getNumber("y_pos", 0)
        heading = self.state_table.getNumber("heading", 0)
        odo.setRobotPos([x, y], heading)

    def updateNetwork(self, odo, ball_list):
        self.setBallTable(ball_list)
        self.getOdometry(odo)