from tdmclient import ClientAsync, aw

class Thymio():
    def __init__(self):
        self.node = None
        self.client = None

    async def connect(self):
        '''Connect to our thymio robot'''
        try:
            self.client = ClientAsync()
            self.node = await self.client.wait_for_node()
            await self.node.lock()
            print("Successfully connected to Thymio robot")
        except Exception as e:
            print(f"Failed to connect to Thymio robot:{e}")
            raise

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