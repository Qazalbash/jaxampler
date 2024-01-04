<!-- Copyright 2023 The Jaxampler Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Jaxampler 🧪 - A JAX-based statistical sampling toolkit

[![Python package](https://github.com/Qazalbash/jaxampler/actions/workflows/python-package.yml/badge.svg)](https://github.com/Qazalbash/jaxampler/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/Qazalbash/jaxampler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Qazalbash/jaxampler/actions/workflows/python-publish.yml)
[![CodeQL](https://github.com/Qazalbash/jaxampler/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/Qazalbash/jaxampler/actions/workflows/github-code-scanning/codeql)
[![Versions](https://img.shields.io/pypi/pyversions/jaxampler.svg)](https://pypi.org/project/jaxampler/)

Jaxampler is a statistical sampling toolkit built on top of JAX. It provides a set of high-performance sampling algorithms for a wide range of statistical distributions. Jaxampler is designed to be easy to use and integrate with existing JAX workflows. It is also designed to be extensible, allowing users to easily add new sampling algorithms and statistical distributions.

Jaxampler is currently in the early stages of development and is not yet ready for production use. However, we welcome contributions from the community to help us improve the library. If you are interested in contributing, please refer to our [contribution guidelines](CONTRIBUTING.md).

## Features

- 🚀 **High-Performance Sampling**: Leverage the power of JAX for high-speed, accurate sampling.
- 🧩 **Versatile Algorithms**: A wide range of sampling methods to suit various applications.
- 🔗 **Easy Integration**: Seamlessly integrates with existing JAX workflows.

## Install

You may install the latest released version of Jaxampler through pip by doing

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

Jaxampler requires Python 3.10 or later. It also requires the following packages:

```bash
jax>=0.4.0
jaxlib>=0.4.0
matplotlib>=3.8.0
tfp-nightly
tqdm
```

The test suite is based on `pytest`. To run the tests, one needs to install pytest and run `pytest` at the root directory of this repo.

## Monte Carlo Methods

Jaxampler currently supports the following Monte Carlo methods:

- [x] Importance Sampling
- [ ] Markov Chain Monte Carlo
- [x] Monte Carlo Box Integration
- [x] Monte Carlo Integration
- [ ] Sequential Monte Carlo
- [ ] Variational Inference
- [ ] Wang-Landau Sampling
- [ ] Worm Algorithm

## Samplers

Jaxampler currently supports the following samplers:

- [x] Accept-Rejection Sampler
- [x] Adaptive Accept-Rejection Sampler
- [ ] Gibbs Sampler
- [ ] Hamiltonian Monte Carlo Sampler
- [x] Hastings Sampler
- [x] Inverse Transform Sampler
- [x] Metropolis-Hastings Sampler
- [ ] Slice Sampler

## Random Variables

Jaxampler currently supports the following random variables:

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
    title   = {{Jaxampler}: A JAX-based statistical sampling toolkit},
    url     = {http://github.com/Qazalbash/jaxampler},
    version = {0.0.4},
    year    = {2023}
}
```

## Contributors

[![List of contributors](https://contrib.rocks/image?repo=Qazalbash/jaxampler)](https://github.com/Qazalbash/jaxampler/graphs/contributors)
