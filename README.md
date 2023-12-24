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

# JAXampler

[![Python package](https://github.com/Qazalbash/jaxampler/actions/workflows/python-package.yml/badge.svg)](https://github.com/Qazalbash/jaxampler/actions/workflows/python-package.yml)
[![Upload Python Package](https://github.com/Qazalbash/jaxampler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Qazalbash/jaxampler/actions/workflows/python-publish.yml)

[JAX](https://jax.readthedocs.io/en/latest/) based lib for sampling statistical distributions.

## Features

- **High-Performance Sampling**: Leverage the power of JAX for high-speed, accurate sampling.
- **Versatile Algorithms**: A wide range of sampling methods to suit various applications.
- **Easy Integration**: Seamlessly integrates with existing JAX workflows.

## Install

Install the latest version of JAXampler using pip:

```bash
pip3 install --upgrade jaxampler
```

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

- [x] [Inverse Transform Sampling](jaxampler/sampler/invtranssampler.py)
- [x] [Accept-Rejection Sampling](jaxampler/sampler/arsampler.py)
- [x] [Adaptive Accept-Rejection Sampling](jaxampler/sampler/aarsampler.py)
- [ ] Metropolis-Hastings
- [ ] Hamiltonian Monte Carlo
- [ ] Slice Sampling
- [ ] Gibbs Sampling
- [ ] Importance Sampling

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
    title   = {{JAXampler}: tool for sampling statistical distributions},
    url     = {http://github.com/Qazalbash/jaxampler},
    version = {0.0.3},
    year    = {2023}
}
```

## Contributing

Contributions are welcome! Please refer to our contribution guidelines for coding standards and pull request processes.

## Contributors

<a href="https://github.com/Qazalbash/jaxampler/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Qazalbash/jaxampler" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
