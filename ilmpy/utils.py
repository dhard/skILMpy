from functools import lru_cache

@lru_cache(maxsize=1_000_000)
def hamming(a: str, b: str) -> int:
    """Compute the Hamming distance between two equal-length strings."""
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))
