import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import numpy as np
import cv2

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

from flybody.fly_envs import flight_imitation_control, flight_imitation
from flybody.utils import display_video
from flybody.tasks.task_utils import retract_wings
from flybody.agents.utils_tf import TestPolicyWrapper


# skeleton
from skeleton import get_skeleton_data

# RL
from RL import Policy, RL, Easy_Policy
from actionTable import ActionTable

# tf for flight_policy
import tensorflow as tf
import tensorflow_probability as tfp
from acme import wrappers

# Baseline pattern for wingbeat pattern generator.
wpg_pattern_path = r'asset\datasets_flight-and-walking-imitation-data\wing_pattern_fmech.npy'
# Flight and walking reference data.
ref_flight_path = r'asset\datasets_flight-and-walking-imitation-data\flight-dataset_saccade-evasion_augmented.hdf5'
ref_walking_path = r'asset\datasets_flight-and-walking-imitation-data\walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5'

flight_policy_path = r'asset\trained-fly-policies\flight'
walk_policy_path = r'asset\trained-fly-policies\walking'
vision_bumps_path = r'asset\trained-fly-policies\vision-bumps'
vision_trench_path = r'asset\trained-fly-policies\vision-trench'

##### code part

# RL & Physics Part

def create_env():
    # Create RL environment.
    env = flight_imitation_control(wpg_pattern_path,
                       ref_flight_path,
                       terminal_com_dist=float('inf'))

    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    timestep = env.reset()
    return env, timestep


def physics_process(shared_queue, lock):
    # Physics process
    env, timestep = create_env()

    # nn
    policy = Easy_Policy(CONFIG['observation_size'], CONFIG['action_size']).to(CONFIG['device'])

    # optimizer only for policy.fc_mean.weight and policy.fc_log_std.weight
    optimizer = optim.Adam(
        [
            {'params': policy.fc_mean.weight},
            {'params': policy.fc_log_std.weight}
        ], 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )

    rl = RL(env, policy, optimizer)

    render_image = env.physics.render(camera_id=1, width=640, height=480)
 
    while True:

        lock.acquire()
        try:
            if shared_queue.full():
                
                obs = shared_queue.get()

                obs = torch.tensor(obs, dtype=torch.float32).to(CONFIG['device'])
                action, log_prob = policy.act(obs)
                action = action.cpu().numpy()

                full_action = np.zeros(11, np.float32)
                full_action[0] = action[0]
                full_action[1] = action[1]

                timestep = env._environment.step(full_action)

                reward = timestep.reward
                rl.get_one_data(log_prob, reward)

                render_image = env.physics.render(camera_id=1, width=640, height=480)
                render_image=cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('render_image', render_image)
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