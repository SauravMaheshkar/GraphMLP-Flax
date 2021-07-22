# Minimal Flax Implementation of GraphMLP

![](https://github.com/SauravMaheshkar/ResMLP-Flax/blob/main/assets/ResMLP%20Banner.png?raw=true)

In the past few months there have been various papers proposing MLP based architectures without Attention or Convolutions In this [paper](https://arxiv.org/abs/2106.04051), titled 'Graph-MLP: Node Classification without Message Passing in Graph', the authors introduce a new architecture for Node Classification.

## Installation

You can install this package from PyPI:

```sh
pip install graphmlp-flax
```

Or directly from GitHub:

```sh
pip install --upgrade git+https://github.com/SauravMaheshkar/GraphMLP-Flax.git
```

## Usage

```python
import jax.numpy as jnp
from graphmlp_flax.models import GMLP

net = GMLP(feature_dim=128, hidden_dim=128, num_classes=10, dtype=jnp.float32)
```

## Development

### 1. Conda Approach

```sh
conda env create --name <env-name> sauravmaheshkar/graphmlp
conda activate <env-name>
```

### 2. Docker Approach

```sh
docker pull ghcr.io/sauravmaheshkar/graphmlp-dev:latest
docker run -it -d --name <container_name> ghcr.io/sauravmaheshkar/graphmlp-dev
```

Use the [Remote Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) Extension in VSCode and [attach to the running container](https://code.visualstudio.com/docs/remote/attach-container). The code resides in the `code/` dir.

Alternatively you can also download the image from [Docker Hub](https://hub.docker.com/r/sauravmaheshkar/graphmlp-dev).

```sh
docker pull sauravmaheshkar/graphmlp-dev
```

## Citations

```bibtex
@misc{hu2021graphmlp,
      title={Graph-MLP: Node Classification without Message Passing in Graph},
      author={Yang Hu and Haoxuan You and Zhecan Wang and Zhicheng Wang and Erjin Zhou and Yue Gao},
      year={2021},
      eprint={2106.04051},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
