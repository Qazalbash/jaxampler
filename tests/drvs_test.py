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

import sys
from itertools import product
from random import randint

from jax import numpy as jnp

sys.path.append("../jaxampler")
from jaxampler.rvs.drvs import Bernoulli, Binomial

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
        assert jnp.all(jnp.abs(rv.cdf(ks[i]) - jnp.sum(rv.pmf(jnp.arange(ks[i] + 1)))) < eps)


def test_Geometric():
    pass
