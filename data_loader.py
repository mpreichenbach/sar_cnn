import numpy as np

def eurosat():
    data = np.load("small_eurosat_data.npy")

    clean = data[0]
    speckle = data[1]
    noisy = data[2]

    del data

    return [clean, speckle, noisy]

def merced():
    data = np.load("synthetic_merced_data.npy")

    clean = data[0]
    speckle = data[1]
    noisy = data[2]

    del data

    return [clean, speckle, noisy]

def merced_64():
    data = np.load("synthetic_merced_64.npy")

    clean = data[0]
    speckle = data[1]
    noisy = data[2]

    del data

    return[clean, speckle, noisy]
