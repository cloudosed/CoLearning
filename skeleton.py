# skeleton.py
# use mmpose

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import math
import time
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

from config import CONFIG


def get_skeleton_data(shared_queue, lock, fps = 30, use_delta = False):
        # 使用模型别名创建推理器
    inferencer = MMPoseInferencer(CONFIG['skeleton_model'], device=CONFIG['device'])
    # result_generator = inferencer(CONFIG['inference_data'], show=False, num_instances = 1, draw_heatmap = True, draw_bbox=True,  return_vis=True)
    result_generator = inferencer('webcam', show=False, num_instances = 1, draw_heatmap = True, draw_bbox=True,  return_vis=True)

    interval = 1 / fps

    
    # Stored_old_data
    # old_data = None

    # old_data = [-150.25997455, -22.67683763, -98.52359478, 192.70287147, -54.4499676, 183.88169509, -172.80983558, 60.68403978, -149.37374408, 103.44381479, -47.9240887, 606.45200289, 14.66227605, 594.14041372,  4.81930072, 575.13195331, 89.59896381, 511.70323299]
    # old_data = np.array(old_data)

    while True:
        start_time = time.time()

        try:
            result = next(result_generator)
        except StopIteration:
            print('Restarting the generator')
            result_generator = inferencer(CONFIG['inference_data'], show=False, num_instances=1, draw_heatmap=True, draw_bbox=True, return_vis=True)
            result = next(result_generator)
        except Exception as e:
            print(f'An error occurred: {e}')
            result_generator = inferencer(CONFIG['inference_data'], show=False, num_instances=1, draw_heatmap=True, draw_bbox=True, return_vis=True)
            result = next(result_generator)

        obs = process_skeleton_data(result)

        if obs is None: continue
        if use_delta and old_data is not None and obs is not None:
            _obs = obs - old_data
            old_data = obs
            obs = _obs

        vis_image = cv2.cvtColor(result['visualization'][0], cv2.COLOR_RGB2BGR)
        vis_image = cv2.resize(vis_image, (640, 480))
        cv2.imshow('vis_image', vis_image)
        cv2.waitKey(1)

        lock.acquire()
        try:
            if shared_queue.full():
                shared_queue.get()
            shared_queue.put(obs)
        finally:
            lock.release()

        elapsed_time = time.time() - start_time
        time_to_wait = max(0, interval - elapsed_time)
        time.sleep(time_to_wait)

def process_skeleton_data(result):
    if result is None or result['predictions'] is None or len(result['predictions']) == 0:
        return None
    keypoints = result['predictions'][0][0]['keypoints']
    scores = result['predictions'][0][0]['keypoint_scores']
    keypoints = np.array(keypoints)
    index = [7,9,8,10,13,14,15,16]
    if scores[5] < 0.1 or scores[6] < 0.1:
        return None
    midpoint = (keypoints[6] + keypoints[5]) / 2
    keypoints = keypoints - midpoint

    obs = []
    for i in index:
        if scores[i] > 0.1:
            if i <= 10:
                obs.append((math.atan(keypoints[i][1] / keypoints[i][0]) + math.pi / 2) / math.pi)
            elif i == 14 or i == 16:
                obs.append(min(max((math.atan(keypoints[i][1] / keypoints[i][0]) + math.pi / 2) / math.pi * 2, 0), 1))
            else:
                obs.append(min(max((math.atan(keypoints[i][0] / keypoints[i][1]) + math.pi / 2) / math.pi * 2 - 1, 0), 1))

        else:
            obs.append(0)
    obs = np.array(obs)

    return obs
