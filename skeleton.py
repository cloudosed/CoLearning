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


def get_skeleton_data(shared_queue, lock, fps = 30):
        # 使用模型别名创建推理器
    inferencer = MMPoseInferencer(CONFIG['skeleton_model'], device=CONFIG['device'])
    # result_generator = inferencer(CONFIG['inference_data'], show=False, num_instances = 1, draw_heatmap = True, draw_bbox=True,  return_vis=True)
    result_generator = inferencer('webcam', show=False, num_instances = 1, draw_heatmap = True, draw_bbox=True,  return_vis=True)

    interval = 1 / fps

    
    # Stored_old_data
    old_data = None

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

        obs = process_skeleton_data(result, old_data)

        if obs is None: continue

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

def process_skeleton_data(result, old_data = None):
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

    if old_data is None:
        old_data = keypoints
        return None
    else:
        obs = []
        for i in index:
            if scores[i] > 0.1:
                obs.append(abs(keypoints[i][0] - old_data[i][0]))
                obs.append(abs(keypoints[i][1] - old_data[i][1]))
            else:
                obs.append(0)
                obs.append(0)
        obs = np.array(obs)
        return obs
