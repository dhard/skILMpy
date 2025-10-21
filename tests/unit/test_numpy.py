import numpy as np

def test_numpy_basic():
    a = np.array([1, 2, 3])
    assert (a * 2).tolist() == [2, 4, 6]
