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
from jaxampler.sampler import AdaptiveAcceptRejectSampler

if __name__ == "__main__":
    scale = 1.35
    N = 10_000

    target_rv = Normal(mu=0.5, sigma=0.2)
    proposal_rv = Beta(alpha=2, beta=2)

    ar_sampler = AdaptiveAcceptRejectSampler()

    start = time()

    samples = ar_sampler.sample(target_rv=target_rv, proposal_rv=proposal_rv, scale=scale, N=N)

    end = time()

    print(len(samples))

    print(f"AcceptRejectSampler: {end - start:.2f}s for {N} samples")

    xx = jnp.linspace(0, 1, N)

    plt.hist(samples, bins=100, density=True, label=f"samples", alpha=0.5, color="brown")
    plt.plot(xx, target_rv.pdf(xx), label=f"target: {target_rv}", color="red")
    plt.plot(xx, scale * proposal_rv.pdf(xx), label=f"proposal: {proposal_rv}", linestyle="--")

    plt.legend()
    plt.tight_layout()
    plt.show()
