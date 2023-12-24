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
