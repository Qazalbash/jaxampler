# Jaxampler

Jaxampler is a tool for sampling statistical distributions using the [JAX](https://jax.readthedocs.io/en/latest/) library.

## Samplers

-   [x] [Inverse Transform Sampling](jaxampler/sampler/invtranssampler.py)
-   [x] [Accept-Rejection Sampling](jaxampler/sampler/arampler.py)
-   [ ] Adaptive Accept-Rejection Sampling
-   [ ] Metropolis-Hastings
-   [ ] Hamiltonian Monte Carlo
-   [ ] Slice Sampling
-   [ ] Gibbs Sampling
-   [ ] Importance Sampling

## Random Variables

### Discrete Random Variables

-   [x] [Bernoulli](jaxampler/rvs/drvs/bernoulli.py)
-   [x] [Binomial](jaxampler/rvs/drvs/binomial.py)
-   [x] [Geometric](jaxampler/rvs/drvs/geometric.py)
-   [x] [Poisson](jaxampler/rvs/drvs/poisson.py)

### Continuous Random Variables

-   [x] [Beta](jaxampler/rvs/crvs/beta.py)
-   [x] [Cauchy](jaxampler/rvs/crvs/cauchy.py)
-   [x] [Chi Square](jaxampler/rvs/crvs/chi2.py)
-   [x] [Exponential](jaxampler/rvs/crvs/exponential.py)
-   [x] [Gamma](jaxampler/rvs/crvs/gamma.py)
-   [x] [Log Normal](jaxampler/rvs/crvs/lognormal.py)
-   [x] [Logistic](jaxampler/rvs/crvs/logistic.py)
-   [x] [Normal](jaxampler/rvs/crvs/normal.py)
-   [x] [Pareto](jaxampler/rvs/crvs/pareto.py)
-   [x] [Rayleigh](jaxampler/rvs/crvs/rayleigh.py)
-   [x] [Student t](jaxampler/rvs/crvs/studentt.py)
-   [x] [Triangular](jaxampler/rvs/crvs/triangular.py)
-   [x] [Truncated Normal](jaxampler/rvs/crvs/truncnormal.py)
-   [x] [Truncated Power Law](jaxampler/rvs/crvs/truncpowerlaw.py)
-   [x] [Uniform](jaxampler/rvs/crvs/uniform.py)
-   [x] [Weibull](jaxampler/rvs/crvs/weibull.py)

## Citing Jaxampler

To cite this repository:

```
@software{jaxampler2023github,
    author  = {Meesum Qazalbash},
    title   = {{JAXampler}: tool for sampling statistical distributions},
    url     = {http://github.com/Qazalbash/jaxampler},
    version = {0.0.1},
    year    = {2023}
}
```

# Contributors

<a href="https://github.com/Qazalbash/jaxampler/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Qazalbash/jaxampler" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
