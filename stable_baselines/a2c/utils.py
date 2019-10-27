import os
from collections import deque

import numpy as np
import tensorflow as tf


def sample(logits):
    """
    Creates a sampling Tensor for non deterministic policies
    when using categorical distribution.
    It uses the Gumbel-max trick: http://amid.fish/humble-gumbel

    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The sampled action
    """
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)


def calc_entropy(logits):
    """
    Calculates the entropy of the output values of the network

    :param logits: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The Entropy of the output values of the network
    """
    # Compute softmax
    a_0 = logits - tf.reduce_max(logits, 1, keepdims=True)
    exp_a_0 = tf.exp(a_0)
    z_0 = tf.reduce_sum(exp_a_0, 1, keepdims=True)
    p_0 = exp_a_0 / z_0
    return tf.reduce_sum(p_0 * (tf.log(z_0) - a_0), 1)


def calc_entropy_softmax(action_proba):
    """
    Calculates the softmax entropy of the output values of the network

    :param action_proba: (TensorFlow Tensor) The input probability for each action
    :return: (TensorFlow Tensor) The softmax entropy of the output values of the network
    """
    return - tf.reduce_sum(action_proba * tf.log(action_proba + 1e-6), axis=1)


def mse(pred, target):
    """
    Returns the Mean squared error between prediction and target

    :param pred: (TensorFlow Tensor) The predicted value
    :param target: (TensorFlow Tensor) The target value
    :return: (TensorFlow Tensor) The Mean squared error between prediction and target
    """
    return tf.reduce_mean(tf.square(pred - target))


def batch_to_seq(tensor_batch, n_batch, n_steps, flat=False):
    """
    Transform a batch of Tensors, into a sequence of Tensors for recurrent policies

    :param tensor_batch: (TensorFlow Tensor) The input tensor to unroll
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_steps: (int) The number of steps to run for each environment
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) sequence of Tensors for recurrent policies
    """
    if flat:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps])
    else:
        tensor_batch = tf.reshape(tensor_batch, [n_batch, n_steps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]


def seq_to_batch(tensor_sequence, flat=False):
    """
    Transform a sequence of Tensors, into a batch of Tensors for recurrent policies

    :param tensor_sequence: (TensorFlow Tensor) The input tensor to batch
    :param flat: (bool) If the input Tensor is flat
    :return: (TensorFlow Tensor) batch of Tensors for recurrent policies
    """
    shape = tensor_sequence[0].get_shape().as_list()
    if not flat:
        assert len(shape) > 1
        n_hidden = tensor_sequence[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, n_hidden])
    else:
        return tf.reshape(tf.stack(values=tensor_sequence, axis=1), [-1])

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


def constant(_):
    """
    Returns a constant value for the Scheduler

    :param _: ignored
    :return: (float) 1
    """
    return 1.


def linear_schedule(progress):
    """
    Returns a linear value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress
    """
    return 1 - progress


def middle_drop(progress):
    """
    Returns a linear value with a drop near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress if (1 - progress) >= 0.75 else 0.075
    """
    eps = 0.75
    if 1 - progress < eps:
        return eps * 0.1
    return 1 - progress


def double_linear_con(progress):
    """
    Returns a linear value (x2) with a flattened tail for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress*2 if (1 - progress*2) >= 0.125 else 0.125
    """
    progress *= 2
    eps = 0.125
    if 1 - progress < eps:
        return eps
    return 1 - progress


def double_middle_drop(progress):
    """
    Returns a linear value with two drops near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) if 0.75 <= 1 - p: 1 - p, if 0.25 <= 1 - p < 0.75: 0.75, if 1 - p < 0.25: 0.125
    """
    eps1 = 0.75
    eps2 = 0.25
    if 1 - progress < eps1:
        if 1 - progress < eps2:
            return eps2 * 0.5
        return eps1 * 0.1
    return 1 - progress


SCHEDULES = {
    'linear': linear_schedule,
    'constant': constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}


class Scheduler(object):
    def __init__(self, initial_value, n_values, schedule):
        """
        Update a value every iteration, with a specific curve

        :param initial_value: (float) initial value
        :param n_values: (int) the total number of iterations
        :param schedule: (function) the curve you wish to follow for your value
        """
        self.step = 0.
        self.initial_value = initial_value
        self.nvalues = n_values
        self.schedule = SCHEDULES[schedule]

    def value(self):
        """
        Update the Scheduler, and return the current value

        :return: (float) the current value
        """
        current_value = self.initial_value * self.schedule(self.step / self.nvalues)
        self.step += 1.
        return current_value

    def value_steps(self, steps):
        """
        Get a value for a given step

        :param steps: (int) The current number of iterations
        :return: (float) the value for the current number of iterations
        """
        return self.initial_value * self.schedule(steps / self.nvalues)


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


def check_shape(tensors, shapes):
    """
    Verifies the tensors match the given shape, will raise an error if the shapes do not match

    :param tensors: ([TensorFlow Tensor]) The tensors that should be checked
    :param shapes: ([list]) The list of shapes for each tensor
    """
    i = 0
    for (tensor, shape) in zip(tensors, shapes):
        assert tensor.get_shape().as_list() == shape, "id " + str(i) + " shape " + str(tensor.get_shape()) + str(shape)
        i += 1


def avg_norm(tensor):
    """
    Return an average of the L2 normalization of the batch

    :param tensor: (TensorFlow Tensor) The input tensor
    :return: (TensorFlow Tensor) Average L2 normalization of the batch
    """
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tensor), axis=-1)))


def gradient_add(grad_1, grad_2, param, verbose=0):
    """
    Sum two gradients

    :param grad_1: (TensorFlow Tensor) The first gradient
    :param grad_2: (TensorFlow Tensor) The second gradient
    :param param: (TensorFlow parameters) The trainable parameters
    :param verbose: (int) verbosity level
    :return: (TensorFlow Tensor) the sum of the gradients
    """
    if verbose > 1:
        print([grad_1, grad_2, param.name])
    if grad_1 is None and grad_2 is None:
        return None
    elif grad_1 is None:
        return grad_2
    elif grad_2 is None:
        return grad_1
    else:
        return grad_1 + grad_2


def q_explained_variance(q_pred, q_true):
    """
    Calculates the explained variance of the Q value

    :param q_pred: (TensorFlow Tensor) The predicted Q value
    :param q_true: (TensorFlow Tensor) The expected Q value
    :return: (TensorFlow Tensor) the explained variance of the Q value
    """
    _, var_y = tf.nn.moments(q_true, axes=[0, 1])
    _, var_pred = tf.nn.moments(q_true - q_pred, axes=[0, 1])
    check_shape([var_y, var_pred], [[]] * 2)
    return 1.0 - (var_pred / var_y)


def total_episode_reward_logger(rew_acc, rewards, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k-1, 0]:dones_idx[k, 0]])
                    summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])

    return rew_acc
