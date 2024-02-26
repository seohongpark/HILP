import numpy as np
from pathlib import Path

import wandb
from matplotlib import figure
from moviepy import editor as mpy
import pathlib


def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    if n_cols is None:
        if v.shape[0] <= 3:
            n_cols = v.shape[0]
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 4
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate(
            (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

    return v


def save_video(label, tensor, fps=15, n_cols=None):
    work_dir = pathlib.Path.cwd()
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t
    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # Encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    plot_path = (work_dir
                 / 'plots'
                 / f'{label}.mp4')
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # clip.write_gif(plot_path, verbose=False, logger=None)
    clip.write_videofile(str(plot_path), fps, audio=False, verbose=False, logger=None)

    # tensor: (t, h, w, c)
    tensor = tensor.transpose(0, 3, 1, 2)
    return wandb.Video(tensor, fps=15, format='mp4')


def record_video(label, renders, n_cols=None, skip_frames=1):
    if len(renders) == 0:
        return
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        render = np.array(render)
        render = np.transpose(render, (0, 3, 1, 2))
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    return save_video(label, renders, n_cols=n_cols)


def get_coord(cfg, env, episode, step):
    if 'domain' in cfg:
        name = cfg.domain
    elif 'task' in cfg:
        name = cfg.task
    else:
        raise Exception()

    if 'ant' in name:
        coord = env.sim.data.qpos.flat[:2]
    elif 'halfcheetah' in name:
        coord = env.sim.data.qpos.flat[:2].copy()
        if episode % 2 == 0:
            y = -(episode // 2)
        else:
            y = (episode + 1) // 2
        coord[1] = y - step / 2000
    elif 'quadruped' in name:
        coord = env.physics.named.data.sensordata['center_of_mass'].copy()
    elif 'jaco' in name:
        coord = env.physics.bind(env.task._hand.tool_center_point).xpos.copy()
    elif 'walker' in name:
        coord = env.physics.named.data.xpos['torso'].copy()
        coord = coord[[0, 2]]
        if episode % 2 == 0:
            y = -(episode // 2)
        else:
            y = (episode + 1) // 2
        coord[1] = y - step / 2000
    elif 'cheetah' in name:
        coord = env.physics.named.data.xpos['torso'].copy()
        coord = coord[[0, 2]]
        if episode % 2 == 0:
            y = -(episode // 2)
        else:
            y = (episode + 1) // 2
        coord[1] = y - step / 2000
    elif 'maze' in name:
        coord = env.state
    else:
        coord = np.array([0., 0.])
    return coord


def extract_state(cfg, env, episode, step):
    if 'domain' in cfg:
        name = cfg.domain
    elif 'task' in cfg:
        name = cfg.task
    else:
        raise Exception()

    if 'ant' in name:
        state = env.unwrapped._get_obs()
    elif 'halfcheetah' in name:
        state = env.get_state()
    else:
        raise NotImplementedError
    return state


class FigManager:
    def __init__(self, label, global_frame, eval_dir, logger):
        self.label = label
        self.global_frame = global_frame
        self.fig = figure.Figure()
        self.ax = self.fig.add_subplot()
        self.eval_dir = eval_dir
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plot_path = (pathlib.Path(self.eval_dir) / 'plots' / f'{self.label}_{self.global_frame}.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(plot_path, dpi=300)
        self.logger.log_figure(self.label, self.fig, self.global_frame)


def plot_trajs(cfg, global_frame, trajs, colors, logger):
    work_dir = Path.cwd()
    with FigManager('TrajPlot', global_frame, work_dir, logger) as fm:
        ax = fm.ax
        square_axis_limit = 0.0
        for trajectory, color in zip(trajs, colors):
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if 'domain' in cfg:
            name = cfg.domain
        elif 'task' in cfg:
            name = cfg.task

        if 'quadruped' in name:
            square_axis_limit = max(15.0, square_axis_limit)
        elif 'walker' in name:
            square_axis_limit = max(30.0, square_axis_limit)
        elif 'ant' in name:
            square_axis_limit = max(40.0, square_axis_limit)

        plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        ax.axis(plot_axis)
        ax.set_aspect('equal')


def plot_phis(cfg, global_frame, points, colors, logger):
    work_dir = Path.cwd()
    with FigManager('PhiPlot', global_frame, work_dir, logger) as fm:
        ax = fm.ax
        square_axis_limit = 2.0
        for point, color in zip(points, colors):
            from matplotlib.patches import Ellipse
            ellipse = Ellipse(xy=point, width=2, height=2, edgecolor=color, lw=1, facecolor='none', alpha=0.8)
            ax.add_patch(ellipse)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(points)))
        square_axis_limit = square_axis_limit * 1.2

        plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        ax.axis(plot_axis)
        ax.set_aspect('equal')


def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((
        colors,
        (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
    ))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors
