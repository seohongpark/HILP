import dataclasses
from collections import OrderedDict, deque
import typing as tp
from typing import Any

from dm_env import Environment
from dm_env import StepType, specs
import numpy as np


from dm_control import suite  # , manipulation
from dm_control.suite.wrappers import action_scale, pixels
from url_benchmark import custom_dmc_tasks as cdmc


S = tp.TypeVar("S", bound="TimeStep")
Env = tp.Union["EnvWrapper", Environment]


@dataclasses.dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: float
    observation: np.ndarray
    physics: np.ndarray = dataclasses.field(default=np.ndarray([]), init=False)

    def first(self) -> bool:
        return self.step_type == StepType.FIRST  # type: ignore

    def mid(self) -> bool:
        return self.step_type == StepType.MID  # type: ignore

    def last(self) -> bool:
        return self.step_type == StepType.LAST  # type: ignore

    def __getitem__(self, attr: str) -> tp.Any:
        return getattr(self, attr)

    def _replace(self: S, **kwargs: tp.Any) -> S:
        for name, val in kwargs.items():
            setattr(self, name, val)
        return self


@dataclasses.dataclass
class ExtendedTimeStep(TimeStep):
    action: tp.Any


class EnvWrapper:
    def __init__(self, env: Env) -> None:
        self._env = env

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if not isinstance(time_step, TimeStep):
            # dm_env time step is a named tuple
            time_step = TimeStep(**time_step._asdict())
        if self.physics is not None:
            return time_step._replace(physics=self.physics.get_state())
        else:
            return time_step

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def observation_spec(self) -> tp.Any:
        assert isinstance(self, EnvWrapper)
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def render(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        return self._env.render(*args, **kwargs)  # type: ignore

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, EnvWrapper):
            return self.base_env
        return env

    @property
    def physics(self) -> tp.Any:
        if hasattr(self._env, "physics"):
            return self._env.physics

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenJacoObservationWrapper(EnvWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for spec in wrapped_obs_spec.values():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((int(np.prod(spec.shape))  # type: ignore
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def observation_spec(self) -> tp.Any:
        return self._obs_spec

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')  # type: ignore
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():  # type: ignore
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)


class ActionRepeatWrapper(EnvWrapper):
    def __init__(self, env: tp.Any, num_repeats: int) -> None:
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action: np.ndarray) -> TimeStep:
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)


class FrameStackWrapper(EnvWrapper):
    def __init__(self, env: Env, num_frames: int, pixels_key: str = 'pixels') -> None:
        super().__init__(env)
        self._num_frames = num_frames
        self._frames: tp.Deque[np.ndarray] = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.Array(np.concatenate([[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0), np.int8, name='observation')

    def observation_spec(self) -> Any:
        return self._obs_spec

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step: TimeStep) -> np.ndarray:
        pixels_ = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels_.shape) == 4:
            pixels_ = pixels_[0]
        return pixels_.transpose(2, 0, 1).copy()

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        pixels_ = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels_)
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        pixels_ = self._extract_pixels(time_step)
        self._frames.append(pixels_)
        return self._augment_time_step(time_step)


class ActionDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def step(self, action) -> Any:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)


class ObservationDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def observation_spec(self) -> Any:
        return self._obs_spec


class ExtendedTimeStepWrapper(EnvWrapper):

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        ts = ExtendedTimeStep(observation=time_step.observation,
                              step_type=time_step.step_type,
                              action=action,
                              reward=time_step.reward or 0.0,
                              discount=time_step.discount or 1.0)
        return super()._augment_time_step(time_step=ts, action=action)


def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed, image_wh=64) -> FlattenJacoObservationWrapper:
    env = cdmc.make_jaco(task, obs_type, seed, image_wh=image_wh)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed, image_wh=64):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=image_wh, width=image_wh, camera_id=camera_id)
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    return env


def make(
    name: str, obs_type='states', frame_stack=1, action_repeat=1, seed=1, image_wh=64,
) -> EnvWrapper:
    assert obs_type in ['states', 'pixels']
    if name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
        _, _, _, task = name.split('_', 3)
    else:
        domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)

    make_fn = _make_jaco if domain == 'jaco' else _make_dmc
    env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed, image_wh=image_wh)

    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)
    return env


def extract_physics(env: Env) -> tp.Dict[str, float]:
    """Extract some physics available in the env"""
    output = {}
    names = ["torso_height", "torso_upright", "horizontal_velocity", "torso_velocity"]
    for name in names:
        if not hasattr(env.physics, name):
            continue
        val: tp.Union[float, np.ndarray] = getattr(env.physics, name)()
        if isinstance(val, (int, float)) or not val.ndim:
            output[name] = float(val)
        else:
            for k, v in enumerate(val):
                output[f"{name}#{k}"] = float(v)
    return output


class FloatStats:
    """Handle for keeping track of the statistics of a float variable"""

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.mean = 0.0
        self._count = 0

    def add(self, value: float) -> "FloatStats":
        self.min = min(value, self.min)
        self.max = max(value, self.max)
        self._count += 1
        self.mean = (self._count - 1) / self._count * self.mean + 1 / self._count * value
        return self

    def items(self) -> tp.Iterator[tp.Tuple[str, float]]:
        for name, val in self.__dict__.items():
            if not name.startswith("_"):
                yield name, val


class PhysicsAggregator:
    """Aggregate stats on the physics of an environment"""

    def __init__(self) -> None:
        self.stats: tp.Dict[str, FloatStats] = {}

    def add(self, env: Env) -> "PhysicsAggregator":
        phy = extract_physics(env)
        for key, val in phy.items():
            self.stats.setdefault(key, FloatStats()).add(val)
        return self

    def dump(self) -> tp.Iterator[tp.Tuple[str, float]]:
        """Exports all statistics and reset the statistics"""
        for key, stats in self.stats.items():
            for stat, val in stats.items():
                yield (f'{key}/{stat}', val)
        self.stats.clear()
