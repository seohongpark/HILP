import logging
import typing as tp
import dataclasses
import collections
from pathlib import Path

import numpy as np
import torch
from dm_env import specs, TimeStep
from tqdm import tqdm

from url_benchmark.dmc import ExtendedTimeStep

Specs = tp.Sequence[specs.Array]
logger = logging.getLogger(__name__)

EpisodeTuple = tp.Tuple[np.ndarray, ...]
Episode = tp.Dict[str, np.ndarray]
T = tp.TypeVar("T", np.ndarray, torch.Tensor)
B = tp.TypeVar("B", bound="EpisodeBatch")


def episode_len(episode: Episode) -> int:
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def load_episode(fn: Path) -> tp.Dict[str, np.ndarray]:
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    return episode  # type: ignore


def relabel_episode(env: tp.Any, episode: tp.Dict[str, np.ndarray]) -> tp.Dict[str, np.ndarray]:
    rewards = []
    states = episode['physics']
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full((1,), reward, dtype=np.float32)
        rewards.append(reward)
    episode['reward'] = np.array(rewards, dtype=np.float32)
    return episode


@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    """For later use
    A container for batchable replayed episodes
    """
    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    meta: tp.Dict[str, T] = dataclasses.field(default_factory=dict)
    _physics: tp.Optional[T] = None
    future_obs: tp.Optional[T] = None

    def __post_init__(self) -> None:
        # some security to be removed later
        assert isinstance(self.reward, (np.ndarray, torch.Tensor))
        assert isinstance(self.discount, (np.ndarray, torch.Tensor))
        assert isinstance(self.meta, dict)

    def to(self, device: str) -> "EpisodeBatch[torch.Tensor]":
        """Creates a new instance on the appropriate device"""
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if field.name == "meta":
                out[field.name] = {x: torch.as_tensor(y, device=device) for x, y in data.items()}  # type: ignore
            elif isinstance(data, (torch.Tensor, np.ndarray)):
                out[field.name] = torch.as_tensor(data, device=device)  # type: ignore
            elif data is None:
                out[field.name] = data
            else:
                raise RuntimeError(f"Not sure what to do with {field.name}: {data}")
        return EpisodeBatch(**out)

    @classmethod
    def collate_fn(cls, batches: tp.List["EpisodeBatch[T]"]) -> "EpisodeBatch[torch.Tensor]":
        """Creates a new instance from several by stacking in a new first dimension
        for all attributes
        """
        out: tp.Dict[str, tp.Any] = {}
        if isinstance(batches[0].obs, np.ndarray):  # move everything to pytorch if first one is numpy
            batches = [b.to("cpu") for b in batches]  # type: ignore
        for field in dataclasses.fields(cls):
            data = [getattr(mf, field.name) for mf in batches]
            # skip fields with None data
            if data[0] is None:
                if any(x is not None for x in data):
                    raise RuntimeError("Found a non-None value mixed with Nones")
                out[field.name] = None
                continue
            # reward and discount can be float which should be converted to
            # tensors for stacking
            if field.name == "meta":
                meta = {k: torch.stack([d[k] for d in data]) for k in data[0]}
                out[field.name] = meta
            elif isinstance(data[0], torch.Tensor):
                out[field.name] = torch.stack(data)
            else:
                raise RuntimeError(f"Not sure what to do with {field.name}: {data}")
                # out[field.name] = [x for y in data for x in y]
        return EpisodeBatch(**out)

    def unpack(self) -> tp.Tuple[T, T, T, T, T]:
        """Unpacks the structure into the legacy unnamed tuple.
        Try to avoid it if possible, this is more likely to be wrong than using names
        """
        return (self.obs, self.action, self.reward, self.discount, self.next_obs)

    def with_no_reward(self: B) -> B:
        reward = self.reward
        reward = torch.zeros_like(reward) if isinstance(reward, torch.Tensor) else 0 * reward
        return dataclasses.replace(self, reward=reward)


class ReplayBuffer:
    def __init__(
            self, max_episodes: int, discount: float, future: float,
            max_episode_length: tp.Optional[int] = None,
            p_currgoal: float = 0., p_randomgoal: float = 0.,
            frame_stack: int = None,
    ) -> None:
        self._max_episodes = max_episodes
        self._discount = discount
        assert 0 <= future <= 1
        self._future = future
        self._p_currgoal = p_currgoal
        self._p_randomgoal = p_randomgoal
        self._current_episode: tp.Dict[str, tp.List[np.ndarray]] = collections.defaultdict(list)
        self._idx = 0
        self._full = False
        self._num_transitions = 0
        self._storage: tp.Dict[str, np.ndarray] = collections.defaultdict()
        self._collected_episodes = 0
        self._batch_names = set(field.name for field in dataclasses.fields(ExtendedTimeStep))
        self._episodes_length = np.zeros(max_episodes, dtype=np.int32)
        self._episodes_selection_probability = None
        self._is_fixed_episode_length = True
        self._max_episode_length = max_episode_length
        self._frame_stack = frame_stack

    def __len__(self) -> int:
        return self._max_episodes if self._full else self._idx

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._backward_compatibility()

    def _backward_compatibility(self):
        if self._storage and not hasattr(self, '_episodes_length'):
            self._episodes_length = np.array([len(array) - 1 for array in self._storage["discount"]], dtype=np.int32)
            self._episodes_length[len(self):] = 0
            assert self._episodes_length[:len(self)].min() == self._episodes_length[:len(self)].max()
            self._episodes_selection_probability = None
            self._is_fixed_episode_length = True
            self._max_episode_length = None

    def add(self, time_step: TimeStep, meta: tp.Mapping[str, np.ndarray]) -> None:
        dtype = np.float32
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for field in dataclasses.fields(time_step):
            value = time_step[field.name]
            if np.isscalar(value):
                value = np.full((1,), value, dtype=dtype)
            if isinstance(value, np.ndarray):
                self._current_episode[field.name].append(np.array(value, dtype=dtype))
        if time_step.last():
            if not hasattr(self, "_batch_names"):
                self._batch_names = set(field.name for field in dataclasses.fields(ExtendedTimeStep))
            for name, value_list in self._current_episode.items():
                values = np.array(value_list, dtype)
                if name not in self._storage:
                    # first iteration, the buffer is created with appropriate size
                    _shape = values.shape
                    if self._max_episode_length is not None:
                        _shape = (self._max_episode_length,) + _shape[1:]
                    self._storage[name] = np.empty((self._max_episodes,) + _shape, dtype=dtype)
                self._storage[name][self._idx][:len(values)] = values
            self._episodes_length[self._idx] = len(self._current_episode['discount']) - 1  # compensate for the dummy transition at the beginning
            if self._episodes_length[self._idx] != self._episodes_length[self._idx - 1] and self._episodes_length[self._idx - 1] != 0:
                self._is_fixed_episode_length = False
            self._current_episode = collections.defaultdict(list)
            self._collected_episodes += 1
            self._idx = (self._idx + 1) % self._max_episodes
            self._full = self._full or self._idx == 0
            self._episodes_selection_probability = None

    @property
    def avg_episode_length(self) -> int:
        return round(self._episodes_length[:len(self)].mean())

    def get_obs(self, storage_key, ep_idx, step_idx):
        if self._frame_stack is None:
            return self._storage[storage_key][ep_idx, step_idx]
        else:
            rets = []
            for i in range(self._frame_stack):
                rets.append(self._storage[storage_key][ep_idx, np.maximum(0, step_idx - (self._frame_stack - i - 1))])
            return np.concatenate(rets, axis=1)

    def sample(self, batch_size, custom_reward: tp.Optional[tp.Any] = None, with_physics: bool = False) -> EpisodeBatch:
        if not hasattr(self, "_batch_names"):
            self._batch_names = set(field.name for field in dataclasses.fields(ExtendedTimeStep))
        if not isinstance(self._future, float):
            assert isinstance(self._future, bool)
            self._future = float(self._future)

        if self._is_fixed_episode_length:
            ep_idx = np.random.randint(0, len(self), size=batch_size)
            random_ep_idx = np.random.randint(0, len(self), size=batch_size)
        else:
            if self._episodes_selection_probability is None:
                self._episodes_selection_probability = self._episodes_length / self._episodes_length.sum()
            ep_idx = np.random.choice(np.arange(len(self._episodes_length)), size=batch_size, p=self._episodes_selection_probability)
            random_ep_idx = np.random.choice(np.arange(len(self._episodes_length)), size=batch_size, p=self._episodes_selection_probability)

        eps_lengths = self._episodes_length[ep_idx]
        random_eps_lengths = self._episodes_length[random_ep_idx]
        # add +1 for the first dummy transition
        step_idx = np.random.randint(0, eps_lengths) + 1
        random_step_idx = np.random.randint(0, random_eps_lengths) + 1
        assert (step_idx <= eps_lengths).all() and (random_step_idx <= random_eps_lengths).all()
        if self._future < 1:
            future_idx = step_idx + np.random.geometric(p=(1 - self._future), size=batch_size)
            future_idx = np.clip(future_idx, 0, eps_lengths)
            assert (future_idx <= eps_lengths).all()
        meta = {name: data[ep_idx, step_idx - 1] for name, data in self._storage.items() if name not in self._batch_names}
        obs = self.get_obs("observation", ep_idx, step_idx - 1)
        action = self._storage['action'][ep_idx, step_idx]
        next_obs = self.get_obs("observation", ep_idx, step_idx)
        phy = self._storage['physics'][ep_idx, step_idx]
        if custom_reward is not None:
            reward = np.array([[custom_reward.from_physics(p)] for p in phy], dtype=np.float32)
        else:
            reward = self._storage['reward'][ep_idx, step_idx]
        discount = self._discount * self._storage['discount'][ep_idx, step_idx]
        future_obs: tp.Optional[np.ndarray] = None
        if self._future < 1:
            future_obs = self.get_obs('observation', ep_idx, future_idx - 1)
            curr_obs = self.get_obs('observation', ep_idx, step_idx - 1)
            future_obs = np.where((np.random.rand(batch_size) < self._p_currgoal / (1. - self._p_randomgoal)).reshape(future_obs.shape[0], *[1] * (len(future_obs.shape) - 1)), curr_obs, future_obs)
            random_obs = self.get_obs('observation', random_ep_idx, random_step_idx - 1)
            future_obs = np.where((np.random.rand(batch_size) < self._p_randomgoal).reshape(future_obs.shape[0], *[1] * (len(future_obs.shape) - 1)), random_obs, future_obs)
        additional = {}
        if with_physics:
            additional["_physics"] = phy
        return EpisodeBatch(obs=obs, action=action, reward=reward, discount=discount,
                            next_obs=next_obs, future_obs=future_obs, meta=meta, **additional)

    def load(self, env: tp.Any, replay_dir: Path, relabel: bool = True) -> None:
        print(f'Loading replay buffer from {replay_dir}')
        eps_fns = sorted(replay_dir.glob('*.npz'))
        for eps_fn in tqdm(eps_fns):
            if self._full:
                break
            episode = load_episode(eps_fn)
            if relabel:
                episode = relabel_episode(env, episode)
            for name, values in episode.items():
                if name not in self._storage:
                    self._storage[name] = np.empty((self._max_episodes,) + values.shape, dtype=np.float32)
                self._storage[name][self._idx] = np.array(values, dtype=np.float32)
            self._episodes_length[self._idx] = len(episode['discount']) - 1  # compensate for the dummy transition at the beginning
            self._idx = (self._idx + 1) % self._max_episodes
            self._full = self._full or self._idx == 0

    def relabel(self, custom_reward) -> None:
        for (ep_idx, phy) in tqdm(enumerate(self._storage["physics"])):
            reward = np.array([[custom_reward.from_physics(p)] for p in phy], dtype=np.float32)
            self._storage["reward"][ep_idx] = reward
        self._max_episodes = len(self._storage["physics"])
        self._full = True
