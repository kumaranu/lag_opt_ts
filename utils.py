import numpy as np


def standardization(x):
    # neighbouring_dists = [np.sqrt((i ** 2).sum()) for i in np.diff(x, axis=0)]
    # mu, std = np.average(neighbouring_dists), np.std(neighbouring_dists)

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)

    # Standardize the vector
    x_stdized = (x - x_mean) / x_std

    return x_stdized