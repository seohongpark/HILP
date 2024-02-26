import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import functools as ft
import matplotlib.gridspec as gridspec
import math


def most_squarelike(n):
    c = int(n ** 0.5)
    while c > 0:
        if n % c in [0, c - 1]:
            return (c, int(math.ceil(n / c)))
        c -= 1


def make_visual(images, metrics, visualization_methods=[]):
    w, h = most_squarelike(len(visualization_methods))
    gs = gridspec.GridSpec(h + 1, w)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    ax = fig.add_subplot(gs[0, :])
    view_images(ax, images, n_images=w * 4)

    for i in range(len(visualization_methods)):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi + 1, wi])
        visualization_methods[i](ax=ax, metrics=metrics)

    plt.tight_layout()
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_image


def make_visual_no_image(metrics, visualization_methods=[]):
    w, h = most_squarelike(len(visualization_methods))
    gs = gridspec.GridSpec(h, w)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    for i in range(len(visualization_methods)):
        wi, hi = i % w, i // w
        ax = fig.add_subplot(gs[hi, wi])
        visualization_methods[i](ax=ax, metrics=metrics)

    plt.tight_layout()
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return out_image


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr


def view_images(ax, images, n_images=4):
    assert len(images.shape) == 4
    assert images.shape[-1] == 3
    interval = images.shape[0] // n_images
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)
    ax.imshow(sel_images)


def visualize_metric(ax, metrics, *, metric_name, linestyle='--', marker='o', **kwargs):
    metric = metrics[metric_name]
    ax.plot(metric, linestyle=linestyle, marker=marker, **kwargs)
    ax.set_ylabel(metric_name)


def visualize_metrics(ax, metrics, *, ylabel=None, metric_names, **kwargs):
    for metric_name in metric_names:
        metric = metrics[metric_name]
        ax.plot(metric, linestyle='--', marker='o', **kwargs)
    ax.set_ylabel(ylabel or metric_names[0])
