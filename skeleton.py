# skeleton.py
# use mmpose

import time
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

from config import CONFIG


def get_skeleton_data(shared_queue, lock, fps = 30):
        # 使用模型别名创建推理器
    inferencer = MMPoseInferencer(CONFIG['skeleton_model'], device=CONFIG['device'])
    result_generator = inferencer('webcam', show=True, num_instances = 1, draw_heatmap = True, draw_bbox=True,  return_vis=True)

    interval = 1 / fps

    while True:
        start_time = time.time()

        result = next(result_generator)

        lock.acquire()
        try:
            if shared_queue.full():
                shared_queue.get()
            shared_queue.put(result)
        finally:
            lock.release()

        elapsed_time = time.time() - start_time
        time_to_wait = max(0, interval - elapsed_time)
        time.sleep(time_to_wait)

# import multiprocessing
# if __name__ == "__main__":
#     # 创建一个 Manager 对象
#     manager = multiprocessing.Manager()
#     # 创建一个共享的 list
#     shared = manager.Queue(1)
#     lock = manager.Lock()

#     # 创建并启动进程
#     process_1 = multiprocessing.Process(target=get_skeleton_data, args=(shared, lock, 30))
#     process_1.start()
#     process_1.join()
