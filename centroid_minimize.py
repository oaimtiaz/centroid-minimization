import numpy as np
from sklearn.preprocessing import MinMaxScaler
from random import random

def calculate_region_center(x_min, x_max, y_min, y_max):
    """
    Calculates the center of a region by averaging the coordinates of the corners.
    """
    return x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2

def calculate_point_centroid(p):
    """
    Calculates the centroid of a set of points by averaging the coordinates.
    """
    return np.mean(p, axis=0)

def dist(p1, p2):
    """
    Calculates the Euclidean distance between two points.
    Wrapper around numpy.linalg.norm.
    """
    return np.linalg.norm(p2 - p1)

def take_samples(p, width, height, x_min=0, y_min=0, n_samples=100, geo_center=None):
    """
    Takes a random sample of points from a region.
    """
    _THRESHOLD = 0.9
    if geo_center is None:
        geo_center = calculate_region_center(x_min, x_min + width, y_min, y_min + height)
    starting_center = calculate_point_centroid(p)
    smallest_dist = dist(starting_center, geo_center)
    samples = None

    # Randomly sample points from the region.
    for i in range(n_samples):
        sampled_vals = []
        for coord in p:
            if random() < _THRESHOLD:
                sampled_vals.append(coord)
        sampled_vals = np.array(sampled_vals)
        sampled_center = calculate_point_centroid(sampled_vals)
        temp_dist = dist(sampled_center, geo_center)
        if temp_dist < smallest_dist:
            smallest_dist = temp_dist
            samples = sampled_vals
        
        if samples is not None and temp_dist == smallest_dist:
            # Keep the set of points with the most points
            if sampled_vals.shape[0] > samples.shape[0]:
                samples = sampled_vals
    return samples


def centroid_minimize(p, width=None, height=None, x_min=0, y_min=0, target_dist=None, max_iterations=100,
                      samples_per_iteration=100, min_samples=None, geo_center=None, **kwargs):
    """
    Perform centroid minimization on a set of points.
    """
    if width is None:
        x_min = np.amin(p[:, 0])
        width = np.amax(p[:, 0]) - x_min
    if height is None:
        y_min = np.amin(p[:, 1])
        height = np.amax(p[:, 1]) - y_min
    if min_samples is None:
        min_samples = p.shape[0] // 40

    if geo_center is None:
        geo_center = calculate_region_center(x_min, x_min + width, y_min, y_min + height)
    point_center = calculate_point_centroid(p)

    if target_dist is None:
        target_dist = dist(point_center, geo_center) / 50

    chosen_sampling = p

    stagnant_count = 0
    for i in range(max_iterations):
        best_sample = take_samples(chosen_sampling, width, height, x_min=x_min, y_min=y_min,
                                   geo_center=geo_center, n_samples=samples_per_iteration)

        if best_sample is not None:
            stagnant_count = 0
            chosen_sampling = best_sample
            new_center = calculate_point_centroid(best_sample)
            new_dist = dist(new_center, geo_center)

            if chosen_sampling.shape[0] <= min_samples or new_dist < target_dist:
                break
        else:
            stagnant_count += 1
            if stagnant_count > 4:
                break

    return chosen_sampling