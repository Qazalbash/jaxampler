from jax.typing import ArrayLike

from .binomial import Binomial


class Bernoulli(Binomial):
    """Bernoulli random variable"""

    def __init__(self, p: ArrayLike, name: str = None) -> None:
        """Initialize the Bernoulli random variable.

        Parameters
        ----------
        p : ArrayLike
            Probability of success.
        name : str, optional
            Name of the random variable, by default None
        """
        super().__init__(p, 1, name)

    def __repr__(self) -> str:
        """Return the string representation of the Bernoulli random variable.

        Returns
        -------
        str
            String representation of the Bernoulli random variable.
        """
        string = f"Bernoulli(p={self._p}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
