from typing import Any, Optional, Union, Callable, Tuple, List

import jax
import jax.numpy as jnp

from flax import linen as nn

Array = Union[jax.Array, Any]
Dtype = Union[jax.typing.DTypeLike, Any]


class ConvStem(nn.Module):
    features: int
    stride: int
    activation: Callable[..., Array] = nn.gelu
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        strides = (self.stride,) * 2
        x = nn.Conv(
                features=self.features,
                kernel_size=(3,3),
                strides=strides,
                padding='SAME',
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype)(x)
        x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = self.activation(x)
        return x


class MBConv(nn.Module):
    features: int
    stride: int
    expand_ratio: float = 4.0
    se_ratio: float = 0.25
    activation: Callable[..., Array] = nn.gelu
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        res = x
        strides = (self.stride,) * 2
        if self.stride > 1:
            res = nn.max_pool(res, window_shape=(3,3), strides=strides, padding='SAME')
            res = nn.Conv(
                    features=self.features,
                    kernel_size=(1,1),
                    strides=(1,1),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype)(res)

        x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        mid_channels = int(x.shape[-1] * self.expand_ratio)

        x = nn.Conv(
                features=mid_channels,
                kernel_size=(1,1),
                strides=strides,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype)(x)
        # From eqn (5), downsampling is applied here
        x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = self.activation(x)

        # Depthwise Convolution
        x = nn.Conv(
                features=mid_channels,
                kernel_size=(3,3),
                strides=(1,1),
                padding='SAME',
                use_bias=False,
                feature_group_count=mid_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype)(x)
        x = nn.BatchNorm(use_running_average=not train, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = self.activation(x)

        # Squeeze and Excitation
        scale = jnp.mean(x, axis=(1,2), keepdims=True) # assuming (batch, height, width, channels)
        sqz_channels = max(1, int(scale.shape[-1] * self.se_ratio))
        scale = nn.Conv(
                    features=sqz_channels,
                    kernel_size=(1,1),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype)(scale)
        scale = self.activation(scale)
        scale = nn.Conv(
                    features=mid_channels,
                    kernel_size=(1,1),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype)(scale)
        scale = nn.sigmoid(x)
        x = x * scale

        x = nn.Conv(
                features=self.features,
                kernel_size=(1,1),
                strides=(1,1),
                dtype=self.dtype,
                param_dtype=self.param_dtype)(x)
        if self.dropout_rate:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        return res + x


def build_relative_position_index(height, width):
    coords = jnp.stack(jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing='ij'))
    coords = coords.reshape((2,-1))
    relative_coords = jnp.expand_dims(coords, -1) - jnp.expand_dims(coords, -2)
    relative_coords = relative_coords.transpose((1, 2, 0))
    relative_coords.at[:, :, 0].add(height - 1)
    relative_coords.at[:, :, 1].add(width - 1)
    relative_coords.at[:, :, 0].multiply(2 * width - 1)
    return relative_coords.sum(-1).reshape((-1))


def build_relative_position_table(rng, height, width, n_heads, param_dtype=jnp.float32):
    initializer = nn.initializers.normal(0.02)
    return initializer(rng, ((2 * height - 1) * (2 * width - 1), n_heads), param_dtype)


class RelativeMultiHeadSelfAttention(nn.Module):
    features: int
    head_dim: int = 32
    n_heads: Optional[int] = None
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        B, H, W, C = x.shape
        assert C % self.head_dim == 0, f"in_channels: {C} must be divisible by head_dim: {self.head_dim}"

        n_heads = self.n_heads if self.n_heads is not None else C // self.head_dim
        hidden_dim = self.head_dim * n_heads

        relative_position_index = self.variable('immutable', 'relative_position_index',
                                                build_relative_position_index, H, W)
        relative_position_table = self.param('relative_position_bias_table', build_relative_position_table,
                                             H, W, n_heads, self.param_dtype)

        x = x.reshape((B, H*W, C))

        q = nn.Dense(hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        k = nn.Dense(hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        v = nn.Dense(hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        q = q.reshape((B, -1, n_heads, self.head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((B, -1, n_heads, self.head_dim)).transpose((0, 2, 3, 1))
        v = v.reshape((B, -1, n_heads, self.head_dim)).transpose((0, 2, 1, 3))

        score = jnp.matmul(q, k) * (hidden_dim ** -0.5)
        pos_bias = jnp.take(relative_position_table, relative_position_index.value, 0)
        pos_bias = pos_bias.reshape(H*W, H*W, -1)
        pos_bias = pos_bias.transpose((2, 0, 1))
        pos_bias = jnp.expand_dims(pos_bias, 0)
        attn = nn.softmax(score + pos_bias, axis=-1)
        if self.dropout_rate:
            attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not train)
        attn = jnp.matmul(attn, v)

        out = attn.transpose(0, 2, 1, 3)
        out = out.reshape((B, -1, hidden_dim))
        out = nn.Dense(features=self.features, dtype=self.dtype, param_dtype=self.param_dtype)(out)

        return out.reshape((B, H, W, -1))


class ConvTransformer(nn.Module):
    features: int
    head_dim: int = 32
    n_heads: Optional[int] = None,
    expand_ratio: float = 4.0
    activation: Callable[..., Array] = nn.gelu
    dropout_rate: float = 0.0
    downsampling: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        proj = x

        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)

        if self.downsampling or x.shape[-1] != self.features:
            if self.downsampling:
                proj = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding='SAME')
                x = nn.max_pool(x, window_shape=(3,3), strides=(2,2), padding='SAME')
            proj = nn.Conv(self.features, kernel_size=(1,1), dtype=self.dtype, param_dtype=self.param_dtype)(proj)

        x = RelativeMultiHeadSelfAttention(
                features=self.features,
                head_dim=self.head_dim,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                param_dtype=self.param_dtype)(x)

        x = x + proj

        out = x
        hidden_dim = int(self.features * self.expand_ratio)

        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.Dense(hidden_dim, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = self.activation(x)
        if self.dropout_rate:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(out.shape[-1], dtype=self.dtype, param_dtype=self.param_dtype)(x)
        if self.dropout_rate:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        return x + out


class CoAtNet(nn.Module):
    stage_configs: List[Any]
    num_classes: int = 1000
    activation: Callable[..., Array] = nn.gelu
    expand_ratio: float = 4.0
    se_ratio: float = 0.25
    dropout_rate: float = 0.0
    head_dim: int = 32
    n_heads: Optional[int] = None
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, train: bool = False) -> Array:
        for i, (b_type, n_block, out_channels, downsampling) in enumerate(self.stage_configs):
            for j in range(n_block):
                downsampling = downsampling and j == 0
                stride = 2 if downsampling else 1
                if b_type.upper() == 'C':
                    x = ConvStem(
                            features=out_channels,
                            stride=stride,
                            activation=self.activation,
                            dtype=self.dtype,
                            param_dtype=self.param_dtype,
                            name='stage{0}_block{1}'.format(i, j))(x, train)
                elif b_type.upper() == 'M':
                    x = MBConv(
                            features=out_channels,
                            stride=stride,
                            expand_ratio=self.expand_ratio,
                            se_ratio=self.se_ratio,
                            activation=self.activation,
                            dropout_rate=self.dropout_rate,
                            dtype=self.dtype,
                            param_dtype=self.param_dtype,
                            name='stage{0}_block{1}'.format(i, j))(x, train)
                elif b_type.upper() == 'T':
                    x = ConvTransformer(
                            features=out_channels,
                            head_dim=self.head_dim,
                            n_heads=self.n_heads,
                            expand_ratio=self.expand_ratio,
                            activation=self.activation,
                            dropout_rate=self.dropout_rate,
                            downsampling=downsampling,
                            dtype=self.dtype,
                            param_dtype=self.param_dtype,
                            name='stage{0}_block{1}'.format(i, j))(x, train)

        x = jnp.mean(x, axis=(1,2)) # assuming (batch, height, width, channels)
        if self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.num_classes, dtype=self.dtype, param_dtype=self.param_dtype)(x)

        return x


def coatnet_0(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2, 64,  True),
        ('M', 2, 96,  True),
        ('M', 3, 192, True),
        ('T', 5, 384, True),
        ('T', 2, 768, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, **kwargs)

def coatnet_1(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2,  64,  True),
        ('M', 2,  96,  True),
        ('M', 6,  192, True),
        ('T', 14, 384, True),
        ('T', 2,  768, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, **kwargs)


def coatnet_2(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2,  128,  True),
        ('M', 2,  128,  True),
        ('M', 6,  256,  True),
        ('T', 14, 512,  True),
        ('T', 2,  1024, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, **kwargs)


def coatnet_3(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  192,  True),
        ('M', 6,  384,  True),
        ('T', 14, 768,  True),
        ('T', 2,  1536, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, **kwargs)


def coatnet_4(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  192,  True),
        ('M', 12, 384,  True),
        ('T', 28, 768,  True),
        ('T', 2,  1536, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, **kwargs)


def coatnet_5(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  256,  True),
        ('M', 12, 512,  True),
        ('T', 28, 1280, True),
        ('T', 2,  2048, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, head_dim=64, **kwargs)


def coatnet_6(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  192,  True),
        ('M', 4,  384,  True),
        ('M', 8,  768,  True),
        ('T', 42, 1536, False),
        ('T', 2,  2048, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, head_dim=128, **kwargs)


def coatnet_7(num_classes: int, **kwargs):
    stage_configs = [
        ('C', 2,  192,  True),
        ('M', 2,  256,  True),
        ('M', 4,  512,  True),
        ('M', 8,  1024, True),
        ('T', 42, 2048, False),
        ('T', 2,  3072, True),
    ]
    return CoAtNet(stage_configs, num_classes=num_classes, head_dim=128, **kwargs)
