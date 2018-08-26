'''
    Read the data in and format it
'''

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
    files = get_files(folder_abs)
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
    x_axis_neg, x_axis_pos = int(x - (w / 2)) + 1, int(x + (w / 2)) + 1
    y_axis_neg, y_axis_pos = int(y - (h / 2)) + 1, int(y + (h / 2)) + 1
    result = arr[y_axis_neg:y_axis_pos, x_axis_neg:x_axis_pos]
    return result


def colour_slice_section(arr, x, y, w, h):
    x_axis_neg, x_axis_pos = int(x - (w / 2)) + 1, int(x + (w / 2)) + 1
    y_axis_neg, y_axis_pos = int(y - (h / 2)) + 1, int(y + (h / 2)) + 1
    img = np.copy(arr)
    img[y_axis_neg:y_axis_pos, x_axis_neg:x_axis_pos] = 1
    return img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_type = "fire"

    slice_size = (5, 5)

    fire_data = [
        (89, 110), (90, 110), (91, 110),
        (89, 111), (90, 111), (91, 111),
        (89, 112), (90, 112), (91, 112)
    ]

    images = []
    folders = get_folders(data_abs[data_type])
    for folder in folders:
        folder_abs = "{}/{}".format(data_abs[data_type], folder)
        img_arr = get_data_from_folder(folder_abs)
        images.append(img_arr)

    images = np.array(images)
    print(images.shape)

    # Normalize channels
    for i, img in enumerate(images):
        images[i] = normalize_channels(img)

    features = []
    labels = []
    metadata = []

    for i, image in enumerate(images):
        folder_name = folders[i]
        for x, y in fire_data:
            feature = slice_section(image, x, y, *slice_size)
            features.append(feature)
            labels.append([1, 0])
            metadata.append((x, y, folder_name))

        for i in range(9):
            feature, x, y = random_sample(image, *slice_size, fire_data)
            features.append(feature)
            labels.append([0, 1])
            metadata.append((x, y, folder_name))

    features = np.array(features)
    labels = np.array(labels)
    metadata = np.array(metadata)

    print("Features: ", features.shape)
    print("Labels: ", labels.shape)

    np.save("data/features", features)
    np.save("data/labels", labels)
    np.save("data/metadata", metadata)