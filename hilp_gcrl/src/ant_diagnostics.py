"""
Very specific code for debugging the antmaze environment.
"""
import matplotlib

matplotlib.use('Agg')

import gym
import d4rl
import numpy as np
import os.path as osp


class Visualizer:
    def __init__(self, env_name, viz_env, dataset, discount):
        data_path = osp.abspath(osp.join(osp.dirname(__file__), f'../antmaze_aux/{env_name}-aux.npz'))
        print('Attempting to load from: ', data_path)
        data = np.load(data_path)
        self.data = {k: data[k] for k in data}
        self.dataset = dataset
        self.viz_env = viz_env
        self.K = 6

        masked = (self.data['pV'] == -500).astype(np.int64)

        from numba import jit
        @jit(nopython=True)
        def get_bfs(gx, gy):
            bfs_default = -1
            bfs = np.full_like(masked, bfs_default)
            q = [(gx, gy)]
            i = 0
            bfs[gx, gy] = 0
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while i < len(q):
                curx, cury = q[i]
                for dx, dy in dirs:
                    nx = curx + dx
                    ny = cury + dy
                    if masked[nx, ny] == 1 or bfs[nx, ny] != bfs_default:
                        continue
                    bfs[nx, ny] = bfs[curx, cury] + 1
                    q.append((nx, ny))
                i += 1
            bfs = np.where(bfs == bfs_default, 500, bfs)
            bfs = -bfs
            return bfs

        goal_bfs = get_bfs(125, 169).astype(np.float32)

        if discount < 1:
            goal_bfs = -(1 - discount ** (-goal_bfs + 1)) / (1 - discount)

        self.data['goal_bfs'] = goal_bfs

    def get_metrics(self, policy_fn):
        directions = self.get_gradients(policy_fn)
        goods = np.clip(self.is_goods(directions), -2.0, 2.0)

        masks = 1.0 - self.data['masked'][::self.K, ::self.K]

        return {
            'average_advantage': np.mean(goods),
            'pct_aligned': np.mean(goods > 0),
            'masked_average_advantage': np.mean(goods * masks) / np.mean(masks),
            'masked_pct_aligned': np.mean((goods > 0) * masks) / np.mean(masks),
        }

    def is_goods(self, directions):
        X, Y = self.data['X'][::self.K], self.data['Y'][::self.K]
        directions = directions.reshape((len(Y), len(X), 2))
        nY, nX = self.data['V'].shape
        print(X.shape, Y.shape, nX, nY)

        goods = np.zeros(directions.shape[:-1])
        for i in range(len(X)):
            for j in range(len(Y)):
                adv = -float('inf')
                for dist in range(9, 12):
                    d = np.round(directions[j, i] * dist).astype(int)
                    new_adv = self.data['V'][
                                  np.clip(j * self.K + d[1], 0, nY - 1),
                                  np.clip(i * self.K + d[0], 0, nX - 1)
                              ] - self.data['V'][j * self.K, i * self.K]
                    adv = max(adv, new_adv)
                goods[j, i] = adv
        return goods

    def get_gradients(self, policy_fn, N=20):
        X, Y = np.meshgrid(self.data['X'][::self.K], self.data['Y'][::self.K])
        observations = np.array([X.flatten(), Y.flatten()]).T
        base_observation = np.copy(self.dataset['observations'][0])
        base_observations = np.tile(base_observation, (observations.shape[0], 1))
        base_observations[:, :2] = observations

        policies = policy_fn(base_observations)
        directions = policies / (1e-6 + np.linalg.norm(policies, axis=1, keepdims=True))
        return directions

    def get_distances(self, trajs):
        final_points = np.array([trajectory['observation'][-1][:2] for trajectory in trajs])
        final_points = np.stack([final_points[:, 1], final_points[:, 0]], axis=1)
        print(final_points.shape)
        from scipy.interpolate import interpn
        return interpn((self.data['Y'], self.data['X']), self.data['pV'], final_points, method='linear',
                       bounds_error=False, fill_value=-300.0)

    def get_distance_metrics(self, trajs):
        import wandb
        distances = self.get_distances(trajs)
        bins = np.arange(self.data['pV'].min(), self.data['pV'].max(), 20)
        hist = np.histogram(distances, bins)
        metrics = {
            'average_distance': np.mean(distances),
            'pct_within_10': np.mean(distances > -10),
            'pct_within_20': np.mean(distances > -20),
            'median_distance': np.median(distances),
            'dist_hist': wandb.Histogram(np_histogram=hist),
        }
        return metrics
