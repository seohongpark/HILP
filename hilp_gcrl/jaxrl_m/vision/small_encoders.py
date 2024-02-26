"""From https://raw.githubusercontent.com/google/flax/main/examples/ppo/models.py"""

from flax import linen as nn
import jax.numpy as jnp

class AtariEncoder(nn.Module):
  """Class defining the actor-critic model."""

  @nn.compact
  def __call__(self, x):
    """Define the convolutional network architecture.

    Architecture originates from "Human-level control through deep reinforcement
    learning.", Nature 518, no. 7540 (2015): 529-533.
    Note that this is different than the one from  "Playing atari with deep
    reinforcement learning." arxiv.org/abs/1312.5602 (2013)

    Network is used to both estimate policy (logits) and expected state value;
    in other words, hidden layers' params are shared between policy and value
    networks, see e.g.:
    github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py
    """
    dtype = jnp.float32
    x = x.astype(dtype) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    return x

small_configs = {
    'atari': AtariEncoder
}