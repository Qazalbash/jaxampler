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

from time import time

from jax import numpy as jnp
from matplotlib import pyplot as plt

from jaxampler.rvs import *
from jaxampler.sampler import InverseTransformSampler

if __name__ == "__main__":
    N = 100_000

    rv = Beta(alpha=3, beta=2)

    sampler = InverseTransformSampler()

    start = time()

    samples = sampler.sample(rv=rv, N=N)

    end = time()

    print(f"InverseTransformSampler: {end - start:.2f}s for {N} samples")

    xx = jnp.linspace(0, 1, N)

    plt.hist(samples, bins=100, density=True, label=f"samples", alpha=0.5, color="brown")
    plt.plot(xx, rv.pdf(xx), label=f"target: {rv}", color="red")
    plt.legend()
    plt.tight_layout()
    plt.show()
