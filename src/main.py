import cv2
import scipy.io
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d import Axes3D
from helpers import *


def plot_data(data_by_types):
    """ Show a scatter plot of the data by type, the first dimension groups per type """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for group, points in enumerate(data_by_types):
        ax.scatter(points[0], points[1], points[2], label="Group " + str(group))

    plt.legend()
    plt.show()


def find_peak(data, idx, r):
    """
    Define a spherical window at the data point of radius r and computing the mean of the points that lie within the
    window. Then shifts the window to the mean and repeats until convergence (the shift is under some threshold t).

    :param data: n-dimensional dataset containing p points
    :param idx: index of the point we wish to calculate its associated density peak
    :param r: search window radius
    :return: peak point
    """

    point = data[:, idx]
    mean_point, _ = calc_mean_and_basin(data, point, r, 4)

    while pdist(np.array([point, mean_point])) > 0.01:
        point = mean_point
        mean_point, _ = calc_mean_and_basin(data, point, r, 4)

    # print("mean_point", mean_point)
    return np.array(mean_point)


def mean_shift(data, r):
    """
    Calculate mean shift
    :param data: n-dimensional dataset containing p points
    :param r: search window radius
    :return: labels, peaks, data grouped in labels
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
        peak = find_peak(data, i, r)
        if len(peaks) > 0:
            similarity = cdist(np.array(peaks), np.array([peak]))
            similar_peak = np.argwhere(similarity < r / 2)

            if len(similar_peak) > 0:
                labels[i] = similar_peak[0, 0]
                continue
            else:
                print("New peak", peak)
                labels[i] = int(len(peaks))
                peaks.append(peak)
        else:
            print("First peak", peak)
            peaks.append(peak)
            labels[i] = 0

    data_grouped = [data[:, labels == group] for group in range(len(peaks))]

    print("Mean shit without optimization done")

    if np.min(peaks) == 1:
        raise Exception("A pixel has not been assigned a group")

    return labels, np.array(peaks), data_grouped


def find_peak_opt(data, idx, r, c):
    """
    Define a spherical window at the data point of radius r and computing the mean of the points that lie within the
    window. Then shifts the window to the mean and repeats until convergence (the shift is under some threshold t).

    Speed-up techniques are:
        1. When the pick is found, all the points in the radius are also labeled
        2. In the search, all the points in a distance smaller or equal to r/c are also labeled

    :param data: n-dimensional dataset containing p points
    :param idx: index of the point we wish to calculate its associated density peak
    :param r: search window radius
    :param c: window radius for basin of attraction
    :return: mean point, ids of the point inside
    """

    point = data[:, idx]
    mean_point, basin_of_attraction = calc_mean_and_basin(data, point, r, c)

    # Creating a set ensures that the values inside appear only once
    ids_in_region = set(basin_of_attraction)

    while pdist(np.array([point, mean_point])) > 0.01:
        point = mean_point
        mean_point, basin_of_attraction = calc_mean_and_basin(data, point, r, 4)
        ids_in_region.update(basin_of_attraction)

    _, basin_of_attraction = calc_mean_and_basin(data, point, r, 1)
    ids_in_region.update(basin_of_attraction)

    return np.array(mean_point), ids_in_region


def mean_shift_opt(data, r, c):
    """
    Calculate mean shift
    :param data: n-dimensional dataset containing p points
    :param r: search window radius
    :param c: window radius for basin of attraction
    :return: labels, peaks, data grouped in labels
    """

    n_pixels = data.shape[1]

    # The label -1 means that is has no class
    labels = np.zeros(n_pixels) - np.ones(n_pixels)

    # Peaks contains the index of the point, therefore we start all as -1
    # peaks = np.zeros(n_points) - np.ones(n_points)

    peaks = []
    iterations = 0
    for i in range(n_pixels):
        if labels[i] != -1:
            continue

        iterations += 1
        peak, ids_in_region = find_peak_opt(data, i, r, c)
        if len(peaks) > 0:
            similarity = cdist(np.array(peaks), np.array([peak]))
            similar_peak = np.argwhere(similarity < r / 2)

            if len(similar_peak) > 0:
                labels[i] = similar_peak[0, 0]
                labels[list(ids_in_region)] = similar_peak[0, 0]
                continue
            else:
                print("New peak", peak)
                labels[i] = int(len(peaks))
                labels[list(ids_in_region)] = int(len(peaks))
                peaks.append(peak)
        else:
            print("First peak", peak)
            peaks.append(peak)
            labels[i] = 0
            labels[list(ids_in_region)] = 0

    # data_grouped = [data[labels == group, :] for group in range(len(peaks))]
    data_grouped = None
    print("Total number of pixels:", n_pixels, ". Finished after", iterations, "iterations ->",
          round(100 * iterations / n_pixels, 2), "%")

    if np.min(peaks) == 1:
        raise Exception("A pixel has not been assigned a group")

    return labels, np.array(peaks), data_grouped


def debug_algorithm():
    data = scipy.io.loadmat('../pts.mat')['data']
    labels, peaks, data_grouped = mean_shift(data, 2)
    plot_data(data_grouped)


def im_segmentation(im, r, c, make_5d):
    """
    Segmentation applied to an image using mean-shift
    :param im: RGB of the image
    :param r: radius for the segmentation
    :param c: window radius for basin of attraction
    :return: labels, peaks, data grouped
    """
    plt.imshow(im)
    plt.show()

    colors = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    if make_5d:
        colors_5d = np.zeros((colors.shape[0], colors.shape[1], 5))
        for y in range(colors.shape[0]):
            for x in range(colors.shape[1]):
                colors_5d[y, x] = np.append(colors[y, x], [y, x])
        colors = colors_5d

    colors_reshaped = colors.reshape((colors.shape[0] * colors.shape[1], colors.shape[2])).T

    labels, peaks, data_grouped = mean_shift_opt(colors_reshaped, r, c)

    for i, peak in enumerate(peaks):
        peaks[i] = [int(value) for value in peak]

    colors_reshaped = colors_reshaped.T
    for i, pixel in enumerate(colors_reshaped):
        colors_reshaped[i] = peaks[int(labels[i])]

    segmented = colors_reshaped.reshape(colors.shape)[:, :, :3]

    segmented = cv2.cvtColor(segmented, cv2.COLOR_LAB2RGB)

    plt.imshow(segmented)
    plt.title("radius = " + str(r))
    plt.show()

    return labels, peaks, data_grouped


def study_image(img_path, r, c, make_5d):
    # Load image
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    labels, peaks, data_grouped = im_segmentation(img_rgb.copy(), r, c, make_5d)
    plot_clusters_3d(img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], img_rgb.shape[2])), labels, peaks, r)


debug_algorithm()

# study_image('../img/55075.jpg', 50, 2, True)
study_image('../img/181091.jpg', 100, 1, True)
study_image('../img/368078.jpg', 50, 2, False)


# plot_data(data_grouped)
