import pytest

from stable_baselines import bench, logger
# from stable_baselines.deepq import DQN, wrap_atari_dqn, CnnPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari
# import stable_baselines.a2c.run_atari as a2c_atari
# import stable_baselines.acer.run_atari as acer_atari
# import stable_baselines.acktr.run_atari as acktr_atari
# import stable_baselines.ppo1.run_atari as ppo1_atari
# import stable_baselines.ppo2.run_atari as ppo2_atari
# import stable_baselines.trpo_mpi.run_atari as trpo_atari


ENV_ID = 'BreakoutNoFrameskip-v4'
SEED = 3
NUM_TIMESTEPS = 500

@pytest.mark.slow
@pytest.mark.skip(reason="Not supported yet, tf2 migration in progress")
def test_deepq():
    """
    test DeepQ on atari
    """
    logger.configure()
    set_global_seeds(SEED)
    env = make_atari(ENV_ID)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)

    model = DQN(env=env, policy=CnnPolicy, learning_rate=1e-4, buffer_size=10000, exploration_fraction=0.1,
                exploration_final_eps=0.01, train_freq=4, learning_starts=10000, target_network_update_freq=1000,
                gamma=0.99, prioritized_replay=True, prioritized_replay_alpha=0.6)
    model.learn(total_timesteps=NUM_TIMESTEPS)

    env.close()
    del model, env
