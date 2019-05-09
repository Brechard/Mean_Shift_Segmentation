import scipy.io
import numpy as np
import cv2
from skimage import io, color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from helpers import *


def plot_data(data_by_types):
    """ Show a scatter plot of the data by type, the first dimension groups per type """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for group, points in enumerate(data_by_types):
        ax.scatter(points[0], points[1], points[2], label="Group " + str(group))

    plt.legend()
    plt.show()


def find_peak(data, idx, r, speed_up):
    """
    Define a spherical window at the data point of radius r and computing the mean of the points that lie within the
    window. Then shifts the window to the mean and repeats until convergence (the shift is under some threshold t).

    Speed-up techniques are:
        1. When the pick is found, all the points in the radius are also labeled
        2. In the search, all the points in a distance smaller or equal to r/c are also labeled

    :param data: n-dimensional dataset containing p points
    :param idx: index of the point we wish to calculate its associated density peak
    :param r: search window radius
    :param speed_up: Flag to use the speed-up techniques
    :return: data set with the new labels
    """

    point = data[:, idx]
    mean_point, basin_of_attraction = calc_mean_and_basin(data, point, r, 4)

    # Creating a set ensures that the values inside appear only once
    if speed_up:
        ids_in_region = set(basin_of_attraction)
    else:
        ids_in_region = None

    while pdist(np.array([point, mean_point])) > 0.01:
        point = mean_point
        mean_point, basin_of_attraction = calc_mean_and_basin(data, point, r, 4)
        if speed_up:
            ids_in_region.update(basin_of_attraction)

    if speed_up:
        _, basin_of_attraction = calc_mean_and_basin(data, point, r, 1)
        ids_in_region.update(basin_of_attraction)

    # print("mean_point", mean_point)
    return np.array(mean_point), ids_in_region


def mean_shift(data, r, speed_up):
    """

    :param data:
    :param r:
    :return:
    """

    n_points = data.shape[1]
    # The label -1 means that is has no class
    labels = np.zeros(n_points) - np.ones(n_points)

    # Peaks contains the index of the point, therefore we start all as -1
    # peaks = np.zeros(n_points) - np.ones(n_points)

    peaks = []
    iterations = 0
    for i in range(n_points):
        if labels[i] != -1:
            continue

        iterations += 1
        peak, ids_in_region = find_peak(data, i, r, speed_up)
        if len(peaks) > 0:
            similarity = cdist(np.array(peaks), np.array([peak]))
            similar_peak = np.argwhere(similarity < r / 2)

            if len(similar_peak) > 0:
                labels[list(ids_in_region)] = similar_peak[0, 0]
                continue
            else:
                print("New peak", peak)
                peaks.append(peak)
        else:
            print("First peak", peak)
            peaks.append(peak)
    return labels, peaks


def debug_algorithm():
    data = scipy.io.loadmat('../pts.mat')['data']
    labels, peaks = mean_shift(data, 2, True)
    data_grouped = [data[:, labels == group] for group in range(len(peaks))]
    plot_data(data_grouped)


debug_algorithm()
