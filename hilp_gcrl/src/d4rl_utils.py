import d4rl
import gym
import numpy as np

from jaxrl_m.dataset import Dataset


def make_env(env_name: str):
    from jaxrl_m.evaluation import EpisodeMonitor
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env


def get_dataset(env: gym.Env,
                env_name: str,
                clip_to_eps: bool = True,
                eps: float = 1e-5,
                dataset=None,
                filter_terminals=False,
                obs_dtype=np.float32,
                goal_conditioned=True,
                ):
        if dataset is None:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        if goal_conditioned:
            dataset['terminals'][-1] = 1

        if filter_terminals:
            # drop terminal transitions
            non_last_idx = np.nonzero(~dataset['terminals'])[0]
            last_idx = np.nonzero(dataset['terminals'])[0]
            penult_idx = last_idx - 1
            new_dataset = dict()
            for k, v in dataset.items():
                if k == 'terminals':
                    v[penult_idx] = 1
                new_dataset[k] = v[non_last_idx]
            dataset = new_dataset

        if 'antmaze' in env_name:
            dones_float = np.zeros_like(dataset['rewards'])
            traj_ends = np.zeros_like(dataset['rewards'])

            for i in range(len(dones_float) - 1):
                traj_end = (np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6)
                traj_ends[i] = traj_end
                if goal_conditioned:
                    dones_float[i] = int(traj_end)
                else:
                    dones_float[i] = int(traj_end or dataset['terminals'][i] == 1.0)
            dones_float[-1] = 1
            traj_ends[-1] = 1
        else:
            dones_float = dataset['terminals'].copy()
            traj_ends = dataset['terminals'].copy()

        observations = dataset['observations'].astype(obs_dtype)
        next_observations = dataset['next_observations'].astype(obs_dtype)

        if goal_conditioned:
            masks = 1.0 - dones_float
        else:
            masks = 1.0 - dataset['terminals'].astype(np.float32)

        return Dataset.create(
            observations=observations,
            actions=dataset['actions'].astype(np.float32),
            rewards=dataset['rewards'].astype(np.float32),
            masks=masks,
            dones_float=dones_float.astype(np.float32),
            next_observations=next_observations,
            traj_ends=traj_ends,
        )


def get_normalization(dataset):
    returns = []
    ret = 0
    for r, term in zip(dataset['rewards'], dataset['dones_float']):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def normalize_dataset(env_name, dataset):
    if 'antmaze' in env_name:
         return  dataset.copy({'rewards': dataset['rewards']- 1.0})
    else:
        normalizing_factor = get_normalization(dataset)
        dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
        return dataset


def kitchen_render(kitchen_env, wh=64):
    from dm_control.mujoco import engine
    camera = engine.MovableCamera(kitchen_env.sim, wh, wh)
    camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
    img = camera.render()
    return img


