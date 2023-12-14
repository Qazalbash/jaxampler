from .binomial import Binomial


class Bernoulli(Binomial):

    def __init__(self, p: float, name: str = None) -> None:
        super().__init__(p, 1, name)

    def __repr__(self) -> str:
        string = f"Bernoulli(p={self._p}"
        if self._name is not None:
            string += f", name={self._name}"
        return string + ")"
