import os
from collections import deque

import numpy as np
import tensorflow as tf


def discount_with_dones(rewards, dones, gamma):
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = 0  # Return: discounted reward
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]


def make_path(path):
    """
    For a given path, create the folders if they do not exist

    :param path: (str) The path
    :return: (bool) Whether or not it finished correctly
    """
    return os.makedirs(path, exist_ok=True)


class EpisodeStats:
    def __init__(self, n_steps, n_envs):
        """
        Calculates the episode statistics

        :param n_steps: (int) The number of steps to run for each environment
        :param n_envs: (int) The number of environments
        """
        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
        self.len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
        self.rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
        self.n_steps = n_steps
        self.n_envs = n_envs

    def feed(self, rewards, masks):
        """
        Update the latest reward and mask

        :param rewards: ([float]) The new rewards for the new step
        :param masks: ([float]) The new masks for the new step
        """
        rewards = np.reshape(rewards, [self.n_envs, self.n_steps])
        masks = np.reshape(masks, [self.n_envs, self.n_steps])
        for i in range(0, self.n_envs):
            for j in range(0, self.n_steps):
                self.episode_rewards[i].append(rewards[i][j])
                if masks[i][j]:
                    reward_length = len(self.episode_rewards[i])
                    reward_sum = sum(self.episode_rewards[i])
                    self.len_buffer.append(reward_length)
                    self.rewbuffer.append(reward_sum)
                    self.episode_rewards[i] = []

    def mean_length(self):
        """
        Returns the average length of each episode

        :return: (float)
        """
        if self.len_buffer:
            return np.mean(self.len_buffer)
        else:
            return 0  # on the first params dump, no episodes are finished

    def mean_reward(self):
        """
        Returns the average reward of each episode

        :return: (float)
        """
        if self.rewbuffer:
            return np.mean(self.rewbuffer)
        else:
            return 0


# For ACER
def get_by_index(input_tensor, idx):
    """
    Return the input tensor, offset by a certain value

    :param input_tensor: (TensorFlow Tensor) The input tensor
    :param idx: (int) The index offset
    :return: (TensorFlow Tensor) the offset tensor
    """
    assert len(input_tensor.get_shape()) == 2
    assert len(idx.get_shape()) == 1
    idx_flattened = tf.range(0, input_tensor.shape[0]) * input_tensor.shape[1] + idx
    offset_tensor = tf.gather(tf.reshape(input_tensor, [-1]),  # flatten input
                              idx_flattened)  # use flattened indices
    return offset_tensor
