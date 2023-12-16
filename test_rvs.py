from jax import numpy as jnp
from jax import random

from jaxampler import Bernoulli

eps = 1e-6


def test_Bernoulli():
    ps = [0.0, 0.1, 0.5, 0.9, 1.0]

    for p in ps:
        rv = Bernoulli(p)
        assert rv._p == p
        assert rv.rvs().shape == (1,)
        assert rv.rvs(N=10).shape == (10,)
        assert rv.rvs(key=random.PRNGKey(0)).shape == (1,)
        assert jnp.all(jnp.abs(rv.pmf(jnp.array([0, 1])) - jnp.array([1 - p, p])) < eps)
        assert jnp.all(jnp.abs(rv.cdf(jnp.array([0, 1])) - jnp.array([1 - p, 1])) < eps)


# def test_Beta():
#     raise NotImplementedError

# def test_Binomial():
#     raise NotImplementedError

# def test_Cauchy():
#     raise NotImplementedError

# def test_Chi2():
#     raise NotImplementedError

# def test_ContinuousRV():
#     raise NotImplementedError

# def test_DiscreteRV():
#     raise NotImplementedError

# def test_Exponential():
#     raise NotImplementedError

# def test_Gamma():
#     raise NotImplementedError

# def test_GenericRV():
#     raise NotImplementedError

# def test_Geometric():
#     raise NotImplementedError

# def test_Logistic():
#     raise NotImplementedError

# def test_LogNormal():
#     raise NotImplementedError

# def test_Normal():
#     raise NotImplementedError

# def test_Pareto():
#     raise NotImplementedError

# def test_Poisson():
#     raise NotImplementedError

# def test_Rayleigh():
#     raise NotImplementedError

# def test_StudentT():
#     raise NotImplementedError

# def test_Triangular():
#     raise NotImplementedError

# def test_TruncNormal():
#     raise NotImplementedError

# def test_TruncPowerLaw():
#     raise NotImplementedError

# def test_Uniform():
#     raise NotImplementedError

# def test_Weibull():
#     raise NotImplementedError
