from src.d4rl_utils import kitchen_render
from typing import Dict
import jax
import gym
import numpy as np
from collections import defaultdict
import time
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def env_reset(env_name, env, goal_info, base_observation, policy_type):
    observation, done = env.reset(), False
    if policy_type == 'random_skill' and 'antmaze' in env_name:
        observation[:2] = [20, 8]
        env.set_state(observation[:15], observation[15:])

    if 'antmaze' in env_name:
        goal = env.wrapped_env.target_goal
        obs_goal = np.concatenate([goal, base_observation[-27:]])
    elif 'kitchen' in env_name:
        if 'visual' in env_name:
            observation = kitchen_render(env)
            obs_goal = goal_info['ob']
        else:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
    else:
        raise NotImplementedError

    return observation, obs_goal


def env_step(env_name, env, action):
    if 'antmaze' in env_name:
        next_observation, reward, done, info = env.step(action)
    elif 'kitchen' in env_name:
        next_observation, reward, done, info = env.step(action)
        if 'visual' in env_name:
            next_observation = kitchen_render(env)
        else:
            next_observation = next_observation[:30]
    else:
        raise NotImplementedError

    return next_observation, reward, done, info


def get_frame(env_name, env):
    if 'antmaze' in env_name:
        size = 200
        cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
    elif 'kitchen' in env_name:
        cur_frame = kitchen_render(env, wh=100).transpose(2, 0, 1)
    else:
        raise NotImplementedError
    return cur_frame


def add_episode_info(env_name, env, info, trajectory):
    if 'antmaze' in env_name:
        info['final_dist'] = np.linalg.norm(trajectory['next_observation'][-1][:2] - env.wrapped_env.target_goal)
    elif 'kitchen' in env_name:
        info['success'] = float(info['episode']['return'] == 4.0)
    else:
        raise NotImplementedError


def evaluate_with_trajectories(
        agent, env: gym.Env, goal_info, env_name, num_episodes, base_observation=None, num_video_episodes=0,
        policy_type='goal_skill', planning_info=None,
) -> Dict[str, float]:
    policy_fn = supply_rng(agent.sample_skill_actions)

    if policy_type == 'goal_skill_planning':
        planning_info['examples']['phis'] = np.array(agent.get_phi(planning_info['examples']['observations']))

    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)

        observation, obs_goal = env_reset(env_name, env, goal_info, base_observation, policy_type)
        done = False

        render = []
        step = 0
        skill = None
        while not done:
            policy_obs = observation
            policy_goal = obs_goal

            if policy_type == 'goal_skill':
                phi_obs, phi_goal = agent.get_phi(np.array([policy_obs, policy_goal]))
                skill = (phi_goal - phi_obs) / np.linalg.norm(phi_goal - phi_obs)
                action = policy_fn(observations=policy_obs, skills=skill, temperature=0.)
            elif policy_type == 'goal_skill_planning':
                phi_obs, phi_goal = agent.get_phi(np.array([policy_obs, policy_goal]))

                for k in range(planning_info['num_recursions']):
                    ex_phis = planning_info['examples']['phis']
                    dists_s = np.linalg.norm(ex_phis - phi_obs, axis=-1)
                    dists_g = np.linalg.norm(ex_phis - phi_goal, axis=-1)
                    dists_diff = np.maximum(dists_s, dists_g)
                    way_idxs = dists_diff.argsort()
                    phi_goal = ex_phis[way_idxs[:planning_info['num_knns']]].mean(axis=0)
                way_skill = (phi_goal - phi_obs) / np.linalg.norm(phi_goal - phi_obs)
                action = policy_fn(observations=policy_obs, skills=way_skill, temperature=0.)
            else:
                raise NotImplementedError

            action = np.array(action)
            next_observation, reward, done, info = env_step(env_name, env, action)
            step += 1

            # Render
            if i >= num_episodes and step % 3 == 0:
                cur_frame = get_frame(env_name, env)
                render.append(cur_frame)
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                skill=skill,
                info=info,
            )
            if i < num_episodes:
                add_to(trajectory, transition)
                add_to(stats, flatten(info))
            observation = next_observation
        if i < num_episodes:
            add_episode_info(env_name, env, info, trajectory)
            add_to(stats, flatten(info, parent_key="final"))
            trajectories.append(trajectory)
        else:
            renders.append(np.array(render))

    scalar_stats = {}
    for k, v in stats.items():
        scalar_stats[k] = np.mean(v)
    return scalar_stats, trajectories, renders


class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
