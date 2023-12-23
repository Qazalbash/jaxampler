# Copyright 2023 The JAXampler Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product
from random import randint

from jax import numpy as jnp

from jaxampler.rvs import Bernoulli, Beta, Binomial
from jaxampler.utils import nCr

eps = 1e-3


def test_Bernoulli():
    ps = [0.0, 0.1, 0.5, 0.9, 1.0]

    for p in ps:
        rv = Bernoulli(p)
        assert rv._p == p
        assert rv.rvs().shape == (1,)
        assert rv.rvs(N=10).shape == (10,)
        assert jnp.all(jnp.abs(rv.pmf([0, 1]) - jnp.array([1 - p, p])) < eps)
        assert jnp.all(jnp.abs(rv.cdf([0, 1]) - jnp.array([1 - p, 1])) < eps)


def test_Beta():
    alpha = [1.0, 2.0, 3.0]
    beta = [1.0, 2.0, 3.0]
    for a, b in product(alpha, beta):
        rv = Beta(alpha=a, beta=b)
        assert rv._alpha == a
        assert rv._beta == b
        assert rv._name == None
        assert rv.rvs().shape == (1,)
        assert rv.rvs(N=10).shape == (10,)
        assert rv.__repr__() == f"Beta(alpha={a}, beta={b})"


def test_Binomial():
    ps = [0.0, 0.1, 0.5, 0.9, 1.0]
    ns = [2, 5, 10]
    ks = [randint(0, i) for i in ns]

    for p, i in product(ps, list(range(len(ns)))):
        rv = Binomial(p, ns[i])
        assert rv._p == p
        assert rv._n == ns[i]
        assert rv.rvs().shape == (1,)
        assert rv.rvs(N=10).shape == (10,)
        assert jnp.all(
            jnp.abs(
                rv.pmf(ks[i]) -
                jnp.array([nCr(ns[i], ks[i]) * jnp.power(p, ks[i]) * jnp.power(1 - p, ns[i] - ks[i])])) < eps)
        # assert jnp.all(jnp.abs(rv.cdf([0, 1]) - jnp.array([1 - p, 1])) < eps)


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
