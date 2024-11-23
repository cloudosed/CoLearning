# skeleton.py
# use mmpose

import time
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

from config import CONFIG


def get_skeleton_data(shared_queue, lock, fps = 30, use_delta = False):
        # 使用模型别名创建推理器
    inferencer = MMPoseInferencer(CONFIG['skeleton_model'], device=CONFIG['device'])
    # result_generator = inferencer('webcam', show=True, num_instances = 1, draw_heatmap = True, draw_bbox=True,  return_vis=True)
    result_generator = inferencer(CONFIG['inference_data'], show=False, num_instances = 1, draw_heatmap = True, draw_bbox=True,  return_vis=True)

    interval = 1 / fps

    
    # Stored_old_data
    # old_data = None

    old_data = [-150.25997455, -22.67683763, -98.52359478, 192.70287147, -54.4499676, 183.88169509, -172.80983558, 60.68403978, -149.37374408, 103.44381479, -47.9240887, 606.45200289, 14.66227605, 594.14041372,  4.81930072, 575.13195331, 89.59896381, 511.70323299]
    old_data = np.array(old_data)

    while True:
        start_time = time.time()

        try:
            result = next(result_generator)
        except StopIteration:
            print('Restarting the generator')
            result_generator = inferencer(CONFIG['inference_data'], show=True, num_instances=1, draw_heatmap=True, draw_bbox=True, return_vis=True)
            result = next(result_generator)
        except Exception as e:
            print(f'An error occurred: {e}')
            result_generator = inferencer(CONFIG['inference_data'], show=True, num_instances=1, draw_heatmap=True, draw_bbox=True, return_vis=True)
            result = next(result_generator)

        if use_delta:
            obs = process_skeleton_data(result)
            old_data = obs
        else:
            obs = process_skeleton_data(result)

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
    keypoints = result['predictions'][0][0]['keypoints']
    keypoints = np.array(keypoints)

    midpoint = (keypoints[6] + keypoints[5]) / 2
    keypoints = keypoints - midpoint

    obs = []
    for i in [0, 7, 8, 9, 10, 13, 14, 15, 16]:
        obs.append(keypoints[i][0])
        obs.append(keypoints[i][1])
    obs = np.array(obs)

    return obs
