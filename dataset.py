import numpy as np

def make_df(low=0, high=101, size=1000):
    rng = np.random.default_rng(seed=None)

    X1_array = rng.integers(low=low, high=high, size=size)
    X2_array = rng.integers(low=low, high=high, size=size)
    X3_array = rng.integers(low=low, high=high, size=size)
    X4_array = rng.integers(low=low, high=high, size=size)

    df = {"X": [], "Y": []}

    for i in range(size):
        row = [X1_array[i], X2_array[i], X3_array[i], X4_array[i]]
        df["X"].append(row)
        value = X1_array[i] + X2_array[i] - X3_array[i] + X4_array[i]
        df["Y"].append(value)

    return df