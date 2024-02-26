import matplotlib

matplotlib.use('Agg')
from matplotlib import patches

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

import gym
import d4rl
import numpy as np
import functools as ft
import math
from jaxrl_m.dataset import Dataset
import matplotlib.gridspec as gridspec


def get_canvas_image(canvas):
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image


def valid_goal_sampler(self, np_random):
    valid_cells = []
    goal_cells = []

    for i in range(len(self._maze_map)):
        for j in range(len(self._maze_map[0])):
            if self._maze_map[i][j] in [0, 'r', 'g']:
                valid_cells.append((i, j))

    sample_choices = valid_cells
    cell = sample_choices[np_random.choice(len(sample_choices))]
    xy = self._rowcol_to_xy(cell, add_random_noise=True)

    random_x = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
    random_y = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling

    xy = (max(xy[0] + random_x, 0), max(xy[1] + random_y, 0))

    return xy


class GoalReachingAnt(gym.Wrapper):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.env.env._wrapped_env.goal_sampler = ft.partial(valid_goal_sampler, self.env.env.env._wrapped_env)
        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'goal': self.env.observation_space,
        })
        self.action_space = self.env.action_space

    def step(self, action):
        next_obs, r, done, info = self.env.step(action)

        achieved = self.get_xy()
        desired = self.target_goal
        distance = np.linalg.norm(achieved - desired)
        info['x'], info['y'] = achieved
        info['achieved_goal'] = np.array(achieved)
        info['desired_goal'] = np.copy(desired)
        info['success'] = float(distance < 0.5)
        done = 'TimeLimit.truncated' in info

        return self.get_obs(next_obs), r, done, info

    def get_obs(self, obs):
        target_goal = obs.copy()
        target_goal[:2] = self.target_goal
        return dict(observation=obs, goal=target_goal)

    def reset(self):
        obs = self.env.reset()
        return self.get_obs(obs)

    def get_starting_boundary(self):
        self = self.env.env.env
        torso_x, torso_y = self._init_torso_x, self._init_torso_y
        S = self._maze_size_scaling
        return (0 - S / 2 + S - torso_x, 0 - S / 2 + S - torso_y), (
        len(self._maze_map[0]) * S - torso_x - S / 2 - S, len(self._maze_map) * S - torso_y - S / 2 - S)

    def XY(self, n=20):
        bl, tr = self.get_starting_boundary()
        X = np.linspace(bl[0] + 0.04 * (tr[0] - bl[0]), tr[0] - 0.04 * (tr[0] - bl[0]), n)
        Y = np.linspace(bl[1] + 0.04 * (tr[1] - bl[1]), tr[1] - 0.04 * (tr[1] - bl[1]), n)

        X, Y = np.meshgrid(X, Y)
        states = np.array([X.flatten(), Y.flatten()]).T
        return states

    def four_goals(self):
        self = self.env.env.env

        valid_cells = []
        goal_cells = []

        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                if self._maze_map[i][j] in [0, 'r', 'g']:
                    valid_cells.append(self._rowcol_to_xy((i, j), add_random_noise=False))

        goals = []
        goals.append(max(valid_cells, key=lambda x: -x[0] - x[1]))
        goals.append(max(valid_cells, key=lambda x: x[0] - x[1]))
        goals.append(max(valid_cells, key=lambda x: x[0] + x[1]))
        goals.append(max(valid_cells, key=lambda x: -x[0] + x[1]))
        return goals

    def draw(self, ax=None):
        if not ax: ax = plt.gca()
        self = self.env.env.env
        torso_x, torso_y = self._init_torso_x, self._init_torso_y
        S = self._maze_size_scaling
        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                struct = self._maze_map[i][j]
                if struct == 1:
                    rect = patches.Rectangle((j * S - torso_x - S / 2, i * S - torso_y - S / 2), S, S,
                                             linewidth=1, edgecolor='none', facecolor='grey', alpha=1.0)

                    ax.add_patch(rect)
        ax.set_xlim(0 - S / 2 + 0.6 * S - torso_x, len(self._maze_map[0]) * S - torso_x - S / 2 - S * 0.6)
        ax.set_ylim(0 - S / 2 + 0.6 * S - torso_y, len(self._maze_map) * S - torso_y - S / 2 - S * 0.6)
        ax.axis('off')


def get_env_and_dataset(env_name):
    env = GoalReachingAnt(env_name)
    dataset = d4rl.qlearning_dataset(env)
    dataset['masks'] = 1.0 - dataset['terminals']
    dataset['dones_float'] = 1.0 - np.isclose(np.roll(dataset['observations'], -1, axis=0),
                                              dataset['next_observations']).all(-1)
    dataset = Dataset.create(**dataset)
    return env, dataset


def plot_value(env, dataset, value_fn, fig, ax, N=20, random=False, title=None):
    observations = env.XY(n=N)

    if random:
        base_observations = np.copy(dataset['observations'][np.random.choice(dataset.size, len(observations))])
    else:
        base_observation = np.copy(dataset['observations'][0])
        base_observations = np.tile(base_observation, (observations.shape[0], 1))

    base_observations[:, :2] = observations

    values = value_fn(base_observations)

    x, y = observations[:, 0], observations[:, 1]
    x = x.reshape(N, N)
    y = y.reshape(N, N)
    values = values.reshape(N, N)
    mesh = ax.pcolormesh(x, y, values, cmap='viridis')
    env.draw(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mesh, cax=cax, orientation='vertical')

    if title:
        ax.set_title(title)


def plot_policy(env, dataset, policy_fn, fig, ax, N=20, random=False, title=None):
    observations = env.XY(n=N)

    if random:
        base_observations = np.copy(dataset['observations'][np.random.choice(dataset.size, len(observations))])
    else:
        base_observation = np.copy(dataset['observations'][0])
        base_observations = np.tile(base_observation, (observations.shape[0], 1))

    base_observations[:, :2] = observations

    policies = policy_fn(base_observations)

    x, y = observations[:, 0], observations[:, 1]
    x = x.reshape(N, N)
    y = y.reshape(N, N)

    policy_x = policies[:, 0].reshape(N, N)
    policy_y = policies[:, 1].reshape(N, N)
    mesh = ax.quiver(x, y, policy_x, policy_y)
    env.draw(ax)
    if title:
        ax.set_title(title)


def plot_trajectories(env, dataset, trajectories, fig, ax, color_list=None):
    if color_list is None:
        from itertools import cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = cycle(color_cycle)

    for color, trajectory in zip(color_list, trajectories):
        obs = np.array(trajectory['observation'])
        all_x = obs[:, 0]
        all_y = obs[:, 1]
        ax.scatter(all_x, all_y, s=5, c=color, alpha=0.02)
        ax.scatter(all_x[-1], all_y[-1], s=50, c=color, marker='*', alpha=0.3)

    env.draw(ax)


def plot_line_trajectories(env, dataset, trajectories, fig, ax, color_list=None):
    if color_list is None:
        from itertools import cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = cycle(color_cycle)

    for color, trajectory in zip(color_list, trajectories):
        obs = np.array(trajectory['observation'])
        all_x = obs[:, 0]
        all_y = obs[:, 1]
        ax.plot(all_x, all_y, color=color, linewidth=0.7)

    env.draw(ax)


def gc_sampling_adaptor(policy_fn):
    def f(observations, *args, **kwargs):
        return policy_fn(observations['observation'], observations['goal'], *args, **kwargs)

    return f


def trajectory_image(env, dataset, trajectories, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    plot_line_trajectories(env, dataset, trajectories, fig, plt.gca(), **kwargs)

    plt.tight_layout()
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image


def value_image(env, dataset, value_fn):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_value(env, dataset, value_fn, fig, plt.gca())
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image


def most_squarelike(n):
    c = int(n ** 0.5)
    while c > 0:
        if n % c in [0, c - 1]:
            return (c, int(math.ceil(n / c)))
        c -= 1


def make_visual(env, dataset, methods):
    h, w = most_squarelike(len(methods))
    gs = gridspec.GridSpec(h, w)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    for i, method in enumerate(methods):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi, wi])
        method(env, dataset, fig=fig, ax=ax)

    plt.tight_layout()
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image


def gcvalue_image(env, dataset, value_fn):
    base_observation = dataset['observations'][0]

    point1, point2, point3, point4 = env.four_goals()
    point3 = (32.75, 24.75)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    points = [point1, point2, point3, point4]
    for i, point in enumerate(points):
        point = np.array(point)
        ax = fig.add_subplot(2, 2, i + 1)

        goal_observation = base_observation.copy()
        goal_observation[:2] = point

        plot_value(env, dataset, partial(value_fn, goal_observation), fig, ax)

        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(point[0], point[1]))
        ax.scatter(point[0], point[1], s=50, c='red', marker='*')

    image = get_canvas_image(canvas)
    plt.close(fig)
    return image
