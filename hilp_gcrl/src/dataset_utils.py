from jaxrl_m.dataset import Dataset
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax
import ml_collections
import jax.numpy as jnp


def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)
random_crop = jax.jit(random_crop, static_argnames=('padding',))


def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)
batched_random_crop = jax.jit(batched_random_crop, static_argnames=('padding',))


@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    discount: float
    geom_sample: int = 1
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    p_aug: float = None

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)

        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)

        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)

        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)
        return goal_indx

    def sample(self, batch_size: int, indx=None, evaluation=False):
        if indx is None:
            indx = np.random.randint(self.dataset.size - 1, size=batch_size)

        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)

        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['masks'] = (1.0 - success.astype(float))
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        if self.p_aug is not None and not evaluation:
            if np.random.rand() < self.p_aug:
                aug_keys = ['observations', 'next_observations', 'goals']
                padding = 3
                crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
                crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int32)], axis=1)
                for key in aug_keys:
                    batch[key] = jax.tree_map(lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr, batch[key])

        if isinstance(batch['goals'], FrozenDict):
            # Freeze the other observations
            batch['observations'] = freeze(batch['observations'])
            batch['next_observations'] = freeze(batch['next_observations'])

        return batch
