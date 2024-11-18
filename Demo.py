import numpy as np
import cv2
import time

import matplotlib.pyplot as plt

from dm_control import mujoco
from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import enums


import multiprocessing

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import CONFIG

# flybody

import os
import sys
module_path = os.path.abspath(os.path.join('./flybody')) # or the path to your source code
sys.path.insert(0, module_path)

from flybody.fly_envs import walk_on_ball, flight_imitation_easy
from flybody.utils import display_video
from flybody.tasks.task_utils import retract_wings


# skeleton
from skeleton import get_skeleton_data

# RL
from RL import Policy, RL



##### code part

# RL & Physics Part

def create_env():
    # Create RL environment.
    env = flight_imitation_easy()
    timestep = env.reset()
    return env, timestep

def physics_process(shared_queue, lock):
    # Physics process
    env, timestep = create_env()
    action_size = env.action_spec().shape[0]
    observation_size = CONFIG['observation_size']

    # nn
    policy = Policy(observation_size, CONFIG['h_size'], CONFIG['action_size']).to(CONFIG['device'])

    # optimizer
    optimizer = optim.Adam(policy.parameters(), lr=CONFIG['lr'])

    rl = RL(env, policy, optimizer)

    
    while True:

        lock.acquire()
        try:
            if shared_queue.full():
                result = shared_queue.get()
                # print(result['predictions'][0][0]['keypoints'])

                # action = np.random.uniform(-0.3, 0.3, size=action_size)
                keypoints = result['predictions'][0][0]['keypoints']
                obs = np.array(keypoints).flatten()

                obs = torch.tensor(obs, dtype=torch.float32).to(CONFIG['device'])
                action, log_prob = policy.act(obs)
                action = action.cpu().detach().numpy()

                # modify action
                zero_action = np.zeros(action_size)
                zero_action[7] = action[0]
                zero_action[4] = action[1]

                timestep = env.step(zero_action)

                reward = timestep.reward
                rl.get_one_data(log_prob, reward)

                vis_image = cv2.cvtColor(result['visualization'][0], cv2.COLOR_RGB2BGR)
                vis_image = cv2.resize(vis_image, (416, 416))

                render_image = env.physics.render(camera_id=1, width=416, height=416)
                display_image = cv2.hconcat([vis_image, render_image])
                cv2.imshow('display_image', display_image)
                cv2.waitKey(1)

        finally:
            lock.release()



import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # 创建一个 Manager 对象
    manager = multiprocessing.Manager()
    # 创建一个共享的 list
    shared_queue = manager.Queue(1)
    lock = manager.Lock()

    # 创建并启动进程
    process_1 = multiprocessing.Process(target=get_skeleton_data, args=(shared_queue, lock, 30))
    process_2 = multiprocessing.Process(target=physics_process, args=(shared_queue, lock) )
    process_1.start()
    process_2.start()

    process_1.join()
    process_2.join()

    # physics_process(shared, lock)