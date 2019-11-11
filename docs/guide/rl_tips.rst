.. _rl_tips:

======================================
Reinforcement Learning Tips and Tricks
======================================

The aim of this section is to guide you for doing reinforcement learning experiments.
It covers general advice about RL (where to start, which algorithm to choose, how to evaluate an algorithm, ...),
as well as tips and tricks when using a custom environment or implementing an RL algorithm.


General advice when using Reinforcement Learning
================================================

TL;DR
-----

1. Read about RL and Stable Baselines
2. Do quantitative experiments and hyperparameter tuning if needed
3. Evaluate the performance using a separate test env
4. For better performances, augment the budget


Like any other subject, if you want to work with RL, you should first read about it (we have a `ressource page <rl.html>`_ to get you started)
to understand what you are using. We also recommend you to read Stable Baselines (SB) documentation and do the `tutorial <https://github.com/araffin/rl-tutorial-jnrr19>`_.
It covers basic usage and guide you towards more advanced concepts of the library.

Reinforcement Learning differs from other machine learning methods in different manners. One is that the data used to train the agent is collected
through interactions with the environment by the agent itself (compared to supervised learning where you have a fixed dataset for instance).
This dependence can lead to vicious circle: if the agent collects poor quality data (e.g., trajectories with no rewards), then it will not improve and continue to amass
bad trajectories.

This factor, among others, explains that results in RL may vary from one run to another (i.e., only the seed of the pseudo-random generator changes).
That's why you should always do several runs to have quantitative results.

Good results in RL are usually dependent on finding appropriate hyperparameters. Recent alogrithms (PPO, SAC, TD3) normally require little hyperparameter tuning
but don't expect the default ones to work on any environment.

We higly recommend you to take a look at the `RL zoo <https://github.com/araffin/rl-baselines-zoo>`_ (or the original papers) for tuned hyperparameters. Doing automatic hyperparameter optimization (included in the rl zoo)
is also part of the best practices when applying an RL algorithm to a new problem.

You should normalize you input (VecNormalize for PPO2/A2C) and look at commmon preprocessing (e.g. for Atari, frame-stack, ...)


Current Limitations of RL
-------------------------

You have to be aware of the current `limitations <https://www.alexirpan.com/2018/02/14/rl-hard.html>`_ of reinforcement learning.


Model-free RL algorithms (i.e. all the algorithms implemented in SB) are usually *sample inefficient*. They require a lot of samples (sometimes millions) to learn something useful.
That's why most of the successes in RL were achieved on games or in simulation only. For instance, in this `work <https://www.youtube.com/watch?v=aTDkYFZFWug>`_ by ETH Zurich, the ANYmal robot was trained in simulation only before being tested in the real world.

As a general advice, to obtain better performances, you should augment the budget of the agent (number of training timesteps).
How to make them more sample efficient? informative (shaped) reward, SRL, initialize with imitation learning,
reduce the number of parameters (e.g. reduce the observation space, action space by constraining it).


In order to to achieved a desired behavior, expert knowledge is often required to design an adequate reward function.
This *reward engineering* (or *RewArt* as coined by `Freek Stulp <http://www.freekstulp.net/>`_), necessitates several iterations. As a good example of reward shaping,
you can take a look at `Deep Mimic <https://xbpeng.github.io/projects/DeepMimic/index.html>`_ which combines imitation learning and reinforcement learning to learn acrobatic moves.

One last limitation of RL is the instability of training. That is to say, you can observe during training a huge drop in performance.
For instance, this behavior is particularly present in `DDPG`, that's why its extension `TD3` tries to tackle that issue.
Other method, like `TRPO` or `PPO` make use of a *trust region* to solve this problem.


How to evaluate an RL algorithm?
--------------------------------

Evaluation methodology: use a test env, evaluate periodically the agent, and use deterministic agent to remove exploration noise.
Looking at the training curve (episode reward function of the timesteps) is a good proxy but usually underestimate the agent true performance.

`blog post <https://openlab-flowers.inria.fr/t/how-many-random-seeds-should-i-use-statistical-power-analysis-in-deep-reinforcement-learning-experiments/457>`_

We suggest you reading `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_ for a good discussion about RL evaluation.


.. RL for Robotics
.. ---------------
..
.. discrete actions -> not really suited
.. continuous actions -> jitterring, recommended to use a PD
..
.. `SAC and applications <https://arxiv.org/abs/1812.05905>`_
.. `CEM-RL <https://openreview.net/forum?id=BkeU5j0ctQ>`_


Which algorithm should I use?
=============================

Discrete, continuous actions?
Multiprocessed?

Discrete: One process: DQN with extensions (prioritized replay), ACER Multiprocessed: PPO2, A2C, ACKTR, ACER
+ check the hyperparameters in the zoo
MPI: PPO1, TRPO

Continuous: One process: SAC, TD3 with rl zoo params Multiprocess: PPO2, TRPO
MPI: PPO1, TRPO, DDPG

Follow the goal env interface: HER + (SAC/TD3/DDPG/DQN)

Experiments on a real hardware: jitterring (later)


Tips and Tricks when creating a custom environment
==================================================

- link to notebook
- normalized observation space
- normalized action space + symmetric (image from twitter)
- start with shaped reward and simple problem
- debug with random agent to check that it follows the gym interface


Tips and Tricks when implementing an RL algorithm
=================================================

- read the paper several times
- read online implementation (if available)
- careful with the shapes (e.g. broadcast value fn) and the gradient
- tools to debug: explained variance
- first sign of life on a simple problem then try harder and harder (usually need hyperparameter tuning)
- what to monitor: entropy (exploration/exploitation), log std, stability (reduce learning rate, linear schedule)
  exploration pb: augment the noise/ entropy coefficient

what not to monitor: actor loss
