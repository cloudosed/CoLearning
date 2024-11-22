"""Utilities for tensorflow networks and nested data structures."""

import numpy as np
from acme import types
from acme.tf import utils as tf2_utils

import tensorflow as tf


class TestPolicyWrapper():
    """At test time, wraps policy to work with non-batched observations.
    Works with distributional policies, e.g. trained with the DMPO agent."""

    def __init__(self, policy, sample=False):
        """
        Args:
            policy: Test policy, e.g. trained policy loaded as 
                policy = tf.saved_model.load('path/to/snapshot').
            sample: Whether to return sample or mean of the distribution.
        """
        self._policy = policy
        self._sample = sample

    def __call__(self, observation: types.NestedArray) -> np.ndarray:
        # Add a dummy batch dimension and as a side effect convert numpy to TF,
        # batched_observation: types.NestedTensor.

       # 将观测值中的所有张量转换为 float32，并添加批次维度
        batched_observation = tf.nest.map_structure(lambda x: tf.cast(tf.expand_dims(x, axis=0), tf.float32), observation)
        # 调用模型，传入字典形式的观测值
        distribution = self._policy(batched_observation)


        if self._sample:
            action = distribution.sample()
        else:
            action = distribution.mean()
        action = action[0, :].numpy()  # Remove batch dimension.
        return action
