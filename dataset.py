import numpy as np


def make_df():
    rng = np.random.default_rng(seed=None) #Only instance of an outside library being used
    #Here I use a dictionary to mimic the structure of a dataframe
    df = {
        "X1": [],
        "X2": [],
        "Y": [],
    }

    X1_array = rng.integers(low=0, high=101, size=1000)
    X2_array = rng.integers(low=0, high=101, size=1000)
    Y_array = []

    for index, X1 in enumerate(X1_array):
        value = X1 + X2_array[index]
        Y_array.append(value)
    df["X1"] = X1_array
    df["X2"] = X2_array
    df["Y"] = Y_array

    return df
