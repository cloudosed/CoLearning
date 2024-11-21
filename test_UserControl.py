import cv2
import numpy as np

# flybody

import os
import sys
module_path = os.path.abspath(os.path.join('./flybody')) # or the path to your source code
sys.path.insert(0, module_path)

from flybody.fly_envs import walk_on_ball, flight_imitation_easy
from flybody.utils import display_video
from flybody.tasks.task_utils import retract_wings

def demo(speed = 0.01):
    env = flight_imitation_easy()
    env.reset()

    stored_angle = 0.03
    flag = 1

    while True:
            render_image = env.physics.render(camera_id=1, width=640, height=480)
            cv2.imshow('display_image', render_image)
            key = cv2.waitKey(1) & 0xFF

            zero_action = np.zeros(12)

            # if stored_angle < 0.125:
            #     stored_angle = stored_angle + speed * flag
            zero_action[4], zero_action[7] = stored_angle, stored_angle

            # if stored_angle >= 0.6 or stored_angle <= -0.2:
            #     flag = -flag

            timestep = env.step(zero_action)
            print(timestep.observation['walker/actuator_activation'])


if __name__ == '__main__':
    demo(0.01)


