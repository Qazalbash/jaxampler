import jax
from jax import numpy as jnp
from jax.scipy.stats import norm
from matplotlib import pyplot as plt

from jaxampler.sampler import MetropolisHastingSampler

if __name__ == "__main__":
    f_pdf = lambda x: 0.5 * (norm.pdf(x, -2.0, 1.0) + norm.pdf(x, 2.0, 1.0))

    step = 0.4
    q_pdf = lambda x1, x2: norm.pdf(x1, x2, step)
    q_rvs = lambda x1, shape, key: jax.random.normal(key, shape=shape, dtype=jnp.float32) * step + x1

    sampler = MetropolisHastingSampler()

    samples = sampler.sample(
        f_pdf=f_pdf,
        q_pdf=q_pdf,
        q_rvs=q_rvs,
        n_chains=5,
        burn_in=25,
        x_curr=jax.random.uniform(jax.random.PRNGKey(90), shape=(5,), minval=-5.0, maxval=5.0),
        N=1000,
        hasting_ratio=True,
    )

    xx = jnp.linspace(-10.0, 10.0, 1000)

    pdf = f_pdf(xx)
    samples = samples.flatten()

    plt.plot(xx, pdf, label="target")
    plt.hist(samples, bins=100, density=True, label="samples")
    plt.legend()
    plt.tight_layout()
    plt.show()
