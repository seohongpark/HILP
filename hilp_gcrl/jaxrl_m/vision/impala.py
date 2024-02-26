import flax.linen as nn
import jax.numpy as jnp

def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

def xavier_init():
    return nn.initializers.xavier_normal()

def kaiming_init():
    return nn.initializers.kaiming_normal()

class ResnetStack(nn.Module):
    num_ch: int
    num_blocks: int
    use_max_pooling: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_ch,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME'
        )(observations)

        if self.use_max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2)
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_ch, kernel_size=(3, 3), strides=1,
                padding='SAME',
                kernel_init=initializer)(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_ch, kernel_size=(3, 3), strides=1,
                padding='SAME', kernel_init=initializer
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    width: int = 1
    use_multiplicative_cond: bool = False
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_ch=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))

        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0
        # x = jnp.reshape(x, (*x.shape[:-2], -1))

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)
            if self.use_multiplicative_cond:
                assert cond_var is not None, "Cond var shouldn't be done when using it"
                print("Using Multiplicative Cond!")
                temp_out = nn.Dense(conv_out.shape[-1], kernel_init=xavier_init())(cond_var)
                x_mult = jnp.expand_dims(jnp.expand_dims(temp_out, 1), 1)
                print ('x_mult shape in IMPALA:', x_mult.shape, conv_out.shape)
                conv_out = conv_out * x_mult

        conv_out = nn.relu(conv_out)
        # print(conv_out.shape, conv_out.reshape((*x.shape[:-3], -1)).shape)
        return conv_out.reshape((*x.shape[:-3], -1))
    
import functools as ft
impala_configs = {
    'impala': ImpalaEncoder,
    'impala_debug': ft.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': ft.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': ft.partial(ImpalaEncoder, stack_sizes=(16, 32, 32, 32)),
    'impala_larger': ft.partial(ImpalaEncoder, stack_sizes=(16, 32, 32, 32, 32)),
    'impala_largest': ft.partial(ImpalaEncoder, stack_sizes=(16, 32, 32, 32, 32, 32)),
    'impala_wider': ft.partial(ImpalaEncoder, width=2),
    'impala_widest': ft.partial(ImpalaEncoder, width=4),
    'impala_deeper': ft.partial(ImpalaEncoder, num_blocks=4),
    'impala_deepest': ft.partial(ImpalaEncoder, num_blocks=8),
}