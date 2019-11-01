.. _rl_tips:

======================================
Reinforcement Learning Tips and Tricks
======================================

General advice when using Reinforcement Learning
================================================

What you should know:

- read about RL
- read about Stable Baselines (we have a tutorial but you should read the doc anyway)
- determinism and dependence to the seed
- hyperparameter tuning -> look at the rl zoo before, or at the papers
- sample efficiency
- normalization (VecNormalize for PPO2/A2C) and preprocessing (frame-stack, ...)
- how to evaluate an agent? (separate test env + link to RL that matters)

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
