# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Cheetah Domain."""

import collections
import os
import typing as tp
from typing import Any, Tuple

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

_DEFAULT_TIME_LIMIT: int
_RUN_SPEED: int
_SPIN_SPEED: int

# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10
_WALK_SPEED = 2
_SPIN_SPEED = 5

SUITE = containers.TaggedTasks()


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward: bool = False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets() -> Tuple[Any, Any]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(
        os.path.join(root_dir, 'custom_dmc_tasks', 'cheetah.xml'))
    return xml, common.ASSETS


@SUITE.add('benchmarking')
def walk(time_limit: int = _DEFAULT_TIME_LIMIT,
         random=None,
         environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=True, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def walk_backward(time_limit: int = _DEFAULT_TIME_LIMIT,
                  random=None,
                  environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=False, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def run_backward(time_limit: int = _DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=False, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def flip(time_limit: int = _DEFAULT_TIME_LIMIT,
         random=None,
         environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=True, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def flip_backward(time_limit: int = _DEFAULT_TIME_LIMIT,
                  random=None,
                  environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=False, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def speed(self) -> Any:
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def angmomentum(self) -> Any:
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Cheetah(base.Task):
    """A `Task` to train a running Cheetah."""

    def __init__(self, move_speed=_RUN_SPEED, forward=True, flip=False, random=None) -> None:
        self._move_speed = move_speed
        self._forward = 1 if forward else -1
        self._flip = flip
        super(Cheetah, self).__init__(random=random)
        self._timeout_progress = 0

    def initialize_episode(self, physics) -> None:
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

        # Stabilize the model before the actual simulation.
        for _ in range(200):
            physics.step()

        physics.data.time = 0
        self._timeout_progress = 0
        super().initialize_episode(physics)

    def get_observation(self, physics) -> tp.Dict[str, Any]:
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics) -> Any:
        """Returns a reward to the agent."""
        if self._flip:
            reward = rewards.tolerance(self._forward * physics.angmomentum(),
                                       bounds=(_SPIN_SPEED, float('inf')),
                                       margin=_SPIN_SPEED,
                                       value_at_margin=0,
                                       sigmoid='linear')

        else:
            reward = rewards.tolerance(self._forward * physics.speed(),
                                       bounds=(self._move_speed, float('inf')),
                                       margin=self._move_speed,
                                       value_at_margin=0,
                                       sigmoid='linear')
        return reward
