import flax.linen as nn
import jax.numpy as jnp

from graphmlp_flax.models import GMLP


def test_instance():

    net = GMLP(feature_dim=128, hidden_dim=128, num_classes=10, dtype=jnp.float32)
    assert isinstance(net, nn.Module)
