import flax.linen as nn
import jax.numpy as jnp

from functools import partial
from typing import Any, Callable, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def xavier_init():
    return nn.initializers.xavier_normal()

def kaiming_init():
    return nn.initializers.kaiming_normal()

ModuleDef = Any


default_kernel_init = nn.initializers.lecun_normal()


class SpatialSoftmax(nn.Module):
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, feature):
        if self.temperature == -1:
            from jax.nn import initializers
            # print("Trainable temperature parameter")
            temperature = self.param('softmax_temperature', initializers.ones, (1), jnp.float32)
        else:
            temperature = 1.

        # print(temperature)
        assert len(feature.shape) == 4
        batch_size, num_featuremaps = feature.shape[0], feature.shape[3]
        feature = feature.transpose(0, 3, 1, 2).reshape(batch_size, num_featuremaps, self.height * self.width)

        softmax_attention = nn.softmax(feature / temperature)
        expected_x = jnp.sum(self.pos_x * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_y = jnp.sum(self.pos_y * softmax_attention, axis=2, keepdims=True).reshape(batch_size, num_featuremaps)
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)

        expected_xy = jnp.reshape(expected_xy, [batch_size, 2*num_featuremaps])
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = default_kernel_init
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, features):
        """ 
        features is B x H x W X C
        """
        kernel = self.param('kernel',
                            self.kernel_init,
                            (self.height, self.width, self.channel, self.num_features),
                            self.param_dtype)
        
        batch_size = features.shape[0]
        assert len(features.shape) == 4
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2))
        features = jnp.reshape(features, [batch_size, -1])
        return features

class MyGroupNorm(nn.GroupNorm):

    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)

class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)




ModuleDef = Any


class MyGroupNorm(nn.GroupNorm):

    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)

class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    norm: str = 'batch'
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    use_multiplicative_cond: bool = False
    use_spatial_learned_embeddings: bool = False
    num_spatial_blocks: int = 8

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = True,
                 cond_var=None):
        
        x = observations.astype(jnp.float32) / 255.0
        # x = jnp.reshape(x, (*x.shape[:-2], -1))

        conv = partial(self.conv, use_bias=False, dtype=self.dtype, kernel_init=kaiming_init())
        if self.norm == 'batch':
            norm = partial(nn.BatchNorm,
                           use_running_average=not train,
                           momentum=0.9,
                           epsilon=1e-5,
                           dtype=self.dtype)
        elif self.norm == 'group':
            norm = partial(MyGroupNorm,
                           num_groups=4,
                           epsilon=1e-5,
                           dtype=self.dtype)
        elif self.norm == 'cross':
            raise NotImplementedError()
            norm = partial(CrossNorm,
                           use_running_average=not train,
                           momentum=0.9,
                           epsilon=1e-5,
                           dtype=self.dtype)
        elif self.norm == 'layer':
            norm = partial(nn.LayerNorm, 
                epsilon=1e-5,
                dtype=self.dtype,
            )
        else:
            raise ValueError('norm not found')

        print('input ', x.shape)
        strides = (2, 2, 2, 1, 1)
        x = conv(self.num_filters, (7, 7), (strides[0], strides[0]),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        print('post conv1', x.shape)

        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(strides[1], strides[1]), padding='SAME')
        print('post maxpool1', x.shape)
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                stride = (strides[i + 1], strides[i + 1]) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=stride,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
                print('post block layer ', x.shape)
                if self.use_multiplicative_cond:
                    assert cond_var is not None, "Cond var is None, nothing to condition on"
                    print("Using Multiplicative Cond!")
                    cond_out = nn.Dense(x.shape[-1], kernel_init=xavier_init())(cond_var)
                    x_mult = jnp.expand_dims(jnp.expand_dims(cond_out, 1), 1)
                    print ('x_mult shape:', x_mult.shape)
                    x = x * x_mult
            print('post block ', x.shape)
            

        if self.use_spatial_learned_embeddings:
            height, width, channel = x.shape[len(x.shape) - 3:]
            print('pre spatial learned embeddings', x.shape)
            x = SpatialLearnedEmbeddings(
                height=height, width=width, channel=channel,
                num_features=self.num_spatial_blocks
            )(x)
            print('post spatial learned embeddings', x.shape)
        elif self.use_spatial_softmax:
            height, width, channel = x.shape[len(x.shape) - 3:]
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1., 1., height),
                jnp.linspace(-1., 1., width)
            )
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            print('pre spatial softmax', x.shape)
            x = SpatialSoftmax(height, width, channel, pos_x, pos_y, self.softmax_temperature)(x)
            print('post spatial softmax', x.shape)
        else:
            x = jnp.mean(x, axis=(len(x.shape) - 3,len(x.shape) - 2))
            print('post flatten', x.shape)
        return x

import functools as ft
resnetv1_configs = {
  'resnetv1-18': ft.partial(ResNetEncoder, stage_sizes=(2, 2, 2, 2),
                   block_cls=ResNetBlock),
  'resnetv1-34': ft.partial(ResNetEncoder, stage_sizes=(3, 4, 6, 3),
                   block_cls=ResNetBlock),
  'resnetv1-50': ft.partial(ResNetEncoder, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock),
  'resnetv1-18-deeper': ft.partial(ResNetEncoder, stage_sizes=(3, 3, 3, 3),
                   block_cls=ResNetBlock),
  'resnetv1-18-deepest': ft.partial(ResNetEncoder, stage_sizes=(4, 4, 4, 4),
                   block_cls=ResNetBlock),
  'resnetv1-18-bridge': ft.partial(ResNetEncoder, stage_sizes=(2, 2, 2, 2),
                   block_cls=ResNetBlock, use_spatial_learned_embeddings=True, num_spatial_blocks=8),
  'resnetv1-34-bridge': ft.partial(ResNetEncoder, stage_sizes=(3, 4, 6, 3),
                   block_cls=ResNetBlock, use_spatial_learned_embeddings=True, num_spatial_blocks=8),
}