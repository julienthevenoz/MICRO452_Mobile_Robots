from tdmclient import ClientAsync, aw
import asyncio

class Thymio():
    def __init__(self, node, client):
        self.node = node
        self.client = client

    @staticmethod
    def motors(left, right):
        return {
            "motor.left.target": [left],
            "motor.right.target": [right],
        }

    def set_motor_speed(self, left, right):
        aw(self.node.set_variables(self.motors(left, right)))

    def stop_robot(self):
        aw(self.node.set_variables(self.motors(0, 0)))

    def read_motors_speed(self):
        aw(self.node.wait_for_variables({"motor.left.speed", "motor.right.speed"}))
        aw(self.client.sleep(0.01))
        speed = [self.node.v.motor.left.speed, self.node.v.motor.right.speed]
        return speed

    def read_prox_sensors(self):
        aw(self.node.wait_for_variables({"prox.horizontal"}))
        aw(self.client.sleep(0.01))
        prox = list(self.node.v.prox.horizontal)
        return prox
