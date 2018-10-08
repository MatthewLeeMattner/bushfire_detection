'''
    Read the data in and format it
'''
import os
from os import listdir
from os.path import isfile, join
from random import randint

import pandas as pd
import numpy as np

from config import data, data_abs


def get_folders(location):
    folders = [f for f in listdir(location) if not isfile(join(location, f))]
    return folders


def get_files(location):
    files = [f for f in listdir(location) if isfile(join(location, f))]
    return files


def load_csv_image(csv_location):
    df = pd.read_csv(csv_location, header=None)
    img_matrix = df.values
    return img_matrix


def get_data_from_folder(location):
    image_matrix = []
    files = get_files(location)
    for file in files:
        if file.split(".")[-1] == "csv" and file.split("_")[-1].split(".")[0] == "data":
            img = load_csv_image("{}/{}".format(location, file))
            image_matrix.append(img)
    image_matrix = np.array(image_matrix)
    image_matrix = np.transpose(image_matrix, (1,2,0))
    return image_matrix


def normalize(arr, min, max):
    return (arr - min)/(max - min)


def normalize_channels(arr):
    for i in range(arr.shape[2]):
        channel = arr[:, :, i]
        channel = normalize(channel, channel.min(), channel.max())
        arr[:, :, i] = channel
    return arr


def random_sample(arr, w, h, not_allowed_pairs):
    '''
    :param not_allowed_pairs: A list of tuples containing (x, y) for areas not to be sampled
    :return:
    '''
    x = randint(20, 160)
    y = randint(20, 200)
    for not_x, not_y in not_allowed_pairs:
        if not_x == x and not_y == y:
            return random_sample(arr, w, h, not_allowed_pairs)
    return slice_section(arr, x, y, w, h), x, y


def slice_section(arr, x, y, w, h):
    x_axis_neg, x_axis_pos = x - int((w / 2)), x + int((w / 2)) + 1
    y_axis_neg, y_axis_pos = y - int((h / 2)), y + int((h / 2)) + 1
    result = arr[y_axis_neg:y_axis_pos, x_axis_neg:x_axis_pos]
    return result


def colour_slice_section(arr, x, y, w, h):
    x_axis_neg, x_axis_pos = int(x - (w / 2)) + 1, int(x + (w / 2))
    y_axis_neg, y_axis_pos = int(y - (h / 2)) + 1, int(y + (h / 2))
    img = np.copy(arr)
    img[y_axis_neg:y_axis_pos, x_axis_neg:x_axis_pos] = 1
    return img


def add_padding(arr, padding=2):
    return np.pad(arr, [(padding, padding), (padding, padding), (0, 0)], mode='constant')


def slice_image(image, w, h, padding=2):
    y_width, x_width, z_width = image.shape
    image = add_padding(image)

    slices = []

    for y in range(y_width):
        for x in range(x_width):
            slice = slice_section(image, x+2, y+2, w, h)
            if slice.shape[0] is not 5:
                print(x, y)
                break

            elif slice.shape[1] is not 5:
                print(x, y)
                break

            slice = np.reshape(slice, (400))
            slices.append(slice)
    slices = np.array(slices)
    slices = np.reshape(slices, (slices.shape[0], 5, 5, 16))
    return slices


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_types = ["maybe_fire"]

    slice_size = (5, 5)

    fire_data = [
        (89, 110), (90, 110), (91, 110),
        (89, 111), (90, 111), (91, 111),
        (89, 112), (90, 112), (91, 112)
    ]

    all_folder = []
    images = []
    for data_type in data_types:
        folders = get_folders(data_abs[data_type])
        for folder in folders:
            folder_abs = "{}/{}".format(data_abs[data_type], folder)
            img_arr = get_data_from_folder(folder_abs)
            images.append(img_arr)
            all_folder.append(folder_abs)

    images = np.array(images)
    print(images.shape)

    # Normalize channels
    for i, img in enumerate(images):
        images[i] = normalize_channels(img)

    features = []
    labels = []
    metadata = []

    for i, image in enumerate(images):
        folder_name = all_folder[i]
        for x, y in fire_data:
            feature = slice_section(image, x, y, *slice_size)
            features.append(feature)
            labels.append([0, 1])
            metadata.append((x, y, folder_name))

        for i in range(9):
            feature, x, y = random_sample(image, *slice_size, fire_data)
            features.append(feature)
            labels.append([1, 0])
            metadata.append((x, y, folder_name))

    features = np.array(features)
    labels = np.array(labels)
    metadata = np.array(metadata)

    if not os.path.exists("data"):
        os.makedirs("data")


    print("Features: ", features.shape)
    print("Labels: ", labels.shape)

    np.save("data/features", features)
    np.save("data/labels", labels)
    np.save("data/metadata", metadata)