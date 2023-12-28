<!-- Copyright 2023 The JAXampler Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# JAXampler 🧪 - A JAX-based statistical sampling toolkit

[![Python package](https://github.com/Qazalbash/jaxampler/actions/workflows/python-package.yml/badge.svg)](https://github.com/Qazalbash/jaxampler/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/Qazalbash/jaxampler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Qazalbash/jaxampler/actions/workflows/python-publish.yml)
[![Versions](https://img.shields.io/pypi/pyversions/jaxampler.svg)](https://pypi.org/project/jaxampler/)

JAXampler is a statistical sampling toolkit built on top of JAX. It provides a set of high-performance sampling algorithms for a wide range of statistical distributions. JAXampler is designed to be easy to use and integrate with existing JAX workflows. It is also designed to be extensible, allowing users to easily add new sampling algorithms and statistical distributions.

JAXampler is currently in the early stages of development and is not yet ready for production use. However, we welcome contributions from the community to help us improve the library. If you are interested in contributing, please refer to our [contribution guidelines](CONTRIBUTING.md).

## Features

- 🚀 **High-Performance Sampling**: Leverage the power of JAX for high-speed, accurate sampling.
- 🧩 **Versatile Algorithms**: A wide range of sampling methods to suit various applications.
- 🔗 **Easy Integration**: Seamlessly integrates with existing JAX workflows.

## Install

You may install the latest released version of JAXampler through pip by doing

```bash
pip3 install --upgrade jaxampler
```

You may install the bleeding edge version by cloning this repo, or doing

```bash
pip3 install --upgrade git+https://github.com/Qazalbash/jaxampler
```

If you would like to take advantage of CUDA, you will additionally need to install a specific version of JAX by doing

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Requirements

JAXampler requires Python 3.10 or later. It also requires the following packages:

```bash
jax>=0.4.0
jaxlib>=0.4.0
matplotlib>=3.8.0
tfp-nightly
tqdm
```

The test suite is based on `pytest`. To run the tests, one needs to install pytest and run `pytest` at the root directory of this repo.

## Samplers

To sample from a distribution, import the corresponding sampler from the `jaxampler.sampler` module and call the `sample` method with the required arguments.

```python
from jaxampler.rvs import Beta, Normal
from jaxampler.sampler import AcceptRejectSampler

scale = 1.35
N = 100_000

target_rv = Normal(mu=0.5, sigma=0.2)
proposal_rv = Beta(alpha=2, beta=2)

ar_sampler = AcceptRejectSampler()

samples = ar_sampler.sample(
    target_rv=target_rv,
    proposal_rv=proposal_rv,
    scale=scale,
    N=N,
)
```

JAXampler currently supports the following samplers:

- [x] Inverse Transform Sampler
- [x] Accept-Rejection Sampler
- [x] Adaptive Accept-Rejection Sampler
- [ ] Metropolis-Hastings Sampler
- [ ] Hamiltonian Monte Carlo Sampler
- [ ] Slice Sampler
- [ ] Gibbs Sampler
- [ ] Importance Sampler

## Random Variables

To create a new random variable, import the corresponding type from the `jaxampler.rvs` i.e. `DiscreteRV` and `ContinuousRV` for discrete and continuous random variables respectively. Then, instantiate the random variable with the required parameters and implement the necessary methods (logpdf, cdf, and ppf etc). JAXampler currently supports the following random variables:

### Discrete Random Variables

- [x] Bernoulli
- [x] Binomial
- [x] Geometric
- [x] Poisson
- [ ] Rademacher

### Continuous Random Variables

- [x] Beta
- [x] Boltzmann
- [x] Cauchy
- [x] Chi
- [x] Exponential
- [x] Gamma
- [ ] Gumbel
- [ ] Laplace
- [x] Log Normal
- [x] Logistic
- [ ] Multivariate Normal
- [x] Normal
- [x] Pareto
- [x] Rayleigh
- [x] Student t
- [x] Triangular
- [x] Truncated Normal
- [x] Truncated Power Law
- [x] Uniform
- [x] Weibull

## Citing Jaxampler

To cite this repository:

```bibtex
@software{jaxampler2023github,
    author  = {Meesum Qazalbash},
    title   = {{JAXampler}: A JAX-based statistical sampling toolkit},
    url     = {http://github.com/Qazalbash/jaxampler},
    version = {0.0.4},
    year    = {2023}
}
```

## Contributors

<a href="https://github.com/Qazalbash/jaxampler/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Qazalbash/jaxampler" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
