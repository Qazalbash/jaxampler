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
    plt.plot(xx, target_rv.pdf_v(xx), label=f"target: {target_rv}", color="red")
    plt.plot(xx, scale * proposal_rv.pdf_v(xx), label=f"proposal: {proposal_rv}", linestyle="--")

    plt.legend()
    plt.tight_layout()
    plt.show()
