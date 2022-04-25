# Code adopted from https://github.com/tomasjakab/keypointgan
import os
import torch
import re
import itertools
import numpy as np
import math
import time

from collections import defaultdict


def swap_points(points, correspondences):
    """
    points: B x N x D
    """
    permutation = list(range((points.shape[1])))
    for a, b in correspondences:
        permutation[a] = b
        permutation[b] = a
    new_points = points[:, permutation, :]
    return new_points

def l2_distance(x, y):
    """
    x: B x N x D
    y: B x N x D
    """
    return torch.sqrt(torch.sum((x - y) ** 2, dim=2))

def mean_l2_distance(x, y):
    """
    x: B x N x D
    y: B x N x D
    """
    return torch.mean(l2_distance(x, y), dim=1)

############################################################################
def format_results_per_activity(results):
    """
    results dict {activity: result}
    """
    s = ''
    for activity, result in sorted(results.items()):
        s += '%s: %.4f ' % (activity, result)
    return s


def format_results_per_activity2(results):
    """
    results dict {activity: result}
    """
    order_full = ['waiting', 'posing', 'greeting', 'directions', 'discussion', 'walking',
            'eating', 'phone_call', 'purchases', 'sitting', 'sitting_down', 'smoking',
            'taking_photo', 'walking_dog', 'walking_together']
    order_yutig = ['Waiting', 'Posing', 'Greeting', 'Directions', 'Discussion', 'Walking']

    if set(order_full) == set(results.keys()):
        order = order_full
    elif set(order_yutig) == set(results.keys()):
        order = order_yutig
    else:
        raise ValueError()

    numbers = ['%.4f' % results[k] for k in order]
    return '\t'.join(order), '\t'.join(numbers)


def mean_per_activity(distances, paths, path_fn):
    activities = [path_fn(p) for p in paths]
    return mean_distance_per_activity(distances, activities)


def human36m_path_to_activity(path):
    return path.split(os.path.sep)[-5]


def y_human36m_path_to_activity(path):
    return re.split('\s|\.', path.split(os.path.sep)[-2])[0]


def mean_distance_per_activity(distances, activities):
    d = defaultdict(list)
    for distance, activity in zip(distances, activities):
        d[activity].append(distance)
    means = {}
    for activity, values in d.items():
        means[activity] = np.mean(values)
    return means


def compute_mean_distance(input, target, correspondeces=None,
                          target_correspondeces=None, used_points=None,
                          offline_prediction=None):
    if target_correspondeces is not None:
        target_swapped = swap_points(target, target_correspondeces)
    else:
        target_swapped = target.clone()
    if correspondeces is not None:
        input_swapped = swap_points(input, correspondeces)
        if offline_prediction is not None:
            offline_prediction_swapped = swap_points(offline_prediction, correspondeces)
    else:
        input_swapped = input.clone()
        if offline_prediction is not None:
            offline_prediction_swapped = offline_prediction.clone()

    if used_points is not None:
        input = input[:, used_points]
        input_swapped = input_swapped[:, used_points]
        if offline_prediction is not None:
            offline_prediction = offline_prediction[:, used_points]
        target = target[:, used_points]
        target_swapped = target_swapped[:, used_points]

    # offline
    if offline_prediction is not None:
        distance = mean_l2_distance(offline_prediction, input)
        swapped_distance = mean_l2_distance(offline_prediction, input_swapped)
        min_idx = distance > swapped_distance
        for i in range(len(min_idx)):
            if min_idx[i]:
                input[i] = input_swapped[i]

    distance = mean_l2_distance(target, input)
    swapped_distance = mean_l2_distance(target_swapped, input)
    correct_flip = distance < swapped_distance
    min_distance = torch.min(distance, swapped_distance)

    return distance, min_distance, correct_flip

def normalize_points(points):
    return (points + 1) / 2.0

def points_to_original(points, height, width, height_ratio, width_ratio):
    """
    points: B x N x 2
    """
    points *= torch.tensor([[[height, width]]], dtype=torch.float32, device=points.device)
    points /= torch.stack([height_ratio, width_ratio], dim=-1, )[:, None].to(points.device)
    return points
