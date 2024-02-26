from gym.envs.registration import register
from d4rl_ext.locomotion import ant
from d4rl_ext.locomotion import maze_env

register(
    id='antmaze-ultra-diverse-v0',
    entry_point='d4rl_ext.locomotion.ant:make_ant_maze_env',
    max_episode_steps=2000,
    kwargs={
        'deprecated': True,
        'maze_map': maze_env.ULTRA_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False,
        'dataset_url':'https://github.com/ZhengyaoJiang/d4rl/releases/download/public/Ant_maze_ultra_noisy_multistart_True_multigoal_True_sparse.hdf5',
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

register(
    id='antmaze-ultra-play-v0',
    entry_point='d4rl_ext.locomotion.ant:make_ant_maze_env',
    max_episode_steps=2000,
    kwargs={
        'deprecated': True,
        'maze_map': maze_env.ULTRA_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False,
        'dataset_url':'https://github.com/ZhengyaoJiang/d4rl/releases/download/public/Ant_maze_ultra_noisy_multistart_True_multigoal_False_sparse.hdf5',
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
    }
)

