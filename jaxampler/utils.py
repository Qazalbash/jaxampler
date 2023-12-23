def nPr(n: int, r: int) -> int:
    """Calculates the number of permutations of `r` objects out of `n`

    Parameters
    ----------
    n : int
        total objects
    r : int
        selected objects

    Returns
    -------
    int
        number of permutations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    fact = [1, 1, 2, 6, 24, 120, 720]
    if n <= len(fact):
        return fact[n] / fact[r] / fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] / fact[n - r]


def nCr(n: int, r: int) -> int:
    """Calculates the number of combinations of `r` objects out of `n`

    Parameters
    ----------
    n : int
        total objects
    r : int
        selected objects

    Returns
    -------
    int
        number of combinations of `r` objects out of `n`
    """
    assert 0 <= r <= n
    fact = [1, 1, 2, 6, 24, 120, 720]
    if n <= len(fact):
        return fact[n] / fact[r] / fact[n - r]
    for i in range(len(fact), n + 1):
        fact.append(fact[i - 1] * i)
    return fact[n] / fact[r] / fact[n - r]
