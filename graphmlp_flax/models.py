from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from chex import Array
from flax import linen as nn

######################################################################
# Utility Functions
######################################################################


def get_feature_dis(x):

    x_dis = x @ x.T
    mask = jnp.eye(jnp.shape(x_dis)[0])
    x_sum = jnp.reshape(jnp.sum(x ** 2, axis=1), newshape=(-1, 1))
    x_sum = jnp.reshape(jnp.sqrt(x_sum), newshape=(-1, 1))
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


######################################################################
# Utility Module
######################################################################


class Mlp(nn.Module):
    hidden_dim: int
    dtype: jnp.float32
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, deterministic=None) -> Array:

        if self.dropout_rate > 0.0:
            deterministic = nn.merge_param(
                "deterministic", self.deterministic, deterministic
            )

        x = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            dtype=self.dtype,
        )(x)
        x = self.act(x)
        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        output = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            dtype=self.dtype,
        )(x)
        return output


######################################################################
# GraphMLP Architecture
######################################################################


class GMLP(nn.Module):
    feature_dim: int
    hidden_dim: int
    num_classes: int
    dtype: jnp.float32
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    act: Callable = nn.gelu

    def setup(self):
        self.mlp = Mlp(
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            act=self.act,
            dtype=self.dtype,
        )
        self.classifier = nn.Dense(features=self.num_classes, dtype=self.dtype)

    @nn.compact
    def __call__(self, x) -> Sequence[Array]:
        x = self.mlp(x)

        feature_cls = x
        Z = x

        if self.training:
            x_dis = get_feature_dis(Z)

        class_feature = self.classifier(feature_cls)
        class_logits = nn.log_softmax(class_feature, axis=1)

        if self.training:
            return class_logits, x_dis
        else:
            return class_logits
