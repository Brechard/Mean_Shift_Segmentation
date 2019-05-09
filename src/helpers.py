import numpy as np
from scipy.spatial.distance import cdist


def get_distances(data, point):
    """ Calculate the distance between the data set and a specific point """
    return cdist(data.T, point.reshape((-1, 1)).T)[:, 0]


def calc_mean_and_basin(data, point, r, c):
    """
    Calculate the mean point in the circle with center "point" and radius "r"
    :param data: n-dimensional dataset containing p points
    :param point: center of the circle of search
    :param r: radius of the circle
    :param c: radius of the circle for the basin of attraction
    :return: mean point, basin of attraction
    """
    distances = get_distances(data, point)
    mean_point = np.mean(data[:, distances < r], axis=1)

    distances_to_mean = get_distances(data, mean_point)
    basin_of_attraction = np.argwhere(distances_to_mean < r / c)[:, 0]

    return mean_point, basin_of_attraction
