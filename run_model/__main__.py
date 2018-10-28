import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
from os import listdir
import numpy as np
import pandas as pd
from os.path import isfile, join

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

def clip_padding(matrix, size=2):
    return matrix[size:-size, size:-size]



def mesh_display(outputs, cmap="coolwarm"):
    outputs = clip_padding(outputs)
    fig = plt.figure(figsize=(40, 10))
    ax = fig.gca(projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xlim3d(0, 222)
    ax.set_ylim3d(0, 181)
    ax.view_init(40, 300)
    # Make data.
    X = range(outputs.shape[0])
    Y = range(outputs.shape[1])
    X, Y = np.meshgrid(X, Y)
    Z = outputs.T

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                           linewidth=0, antialiased=True, shade=True)

    # Customize the z axis.
    ax.set_zlim(-2, 2.00)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)


def heatmap_display(outputs, cmap="coolwarm"):
    fig, ax = plt.subplots(figsize=(10, 12.25))
    ax.set_yticks([])
    ax.set_xticks([])
    sns.heatmap(outputs, cmap=cmap, ax=ax)


def convert_output_to_heatmap(outputs, shape, threshold=None):
    arr = []
    if len(outputs.shape) is not 2:
        for output in outputs:
            arr.append(output)
        arr = np.array(arr)
    else:
        for not_fire, fire in outputs:
            arr.append(fire - not_fire)
        arr = np.array(arr)
        arr = (arr - arr.min()) / (arr.max() - arr.min())

    if threshold is not None:
        for i in range(len(arr)):
            if arr[i] > threshold:
                arr[i] = 1
            else:
                arr[i] = 0

    return np.reshape(arr, shape)


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


X = tf.placeholder(tf.float32, shape=(None, 400), name="features")
y = tf.placeholder(tf.float32, shape=(None, 2), name="labels")

with tf.name_scope("input_layer"):
    x1 = tf.layers.dense(X, units=200)
    a1 = tf.nn.tanh(x1)
with tf.name_scope("hidden_layer"):
    x2 = tf.layers.dense(a1, units=100)
    a2 = tf.nn.tanh(x2)
with tf.name_scope("output_layer"):
    x3 = tf.layers.dense(a2, units=2, name="output")


saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "model/ann.ckpt")


folder_abs = sys.argv[1]
img_arr = get_data_from_folder(folder_abs)
img_arr = normalize_channels(img_arr)
img_arr = slice_image(img_arr, 5, 5)
img_arr = np.nan_to_num(img_arr)
img_arr = np.reshape(img_arr, (img_arr.shape[0], img_arr.shape[1] * img_arr.shape[2] * img_arr.shape[3]))
outputs = sess.run(x3, feed_dict={
            X: img_arr
        })

outputs = softmax(outputs, axis=1)

cmap = "coolwarm"

heatmap_data = convert_output_to_heatmap(outputs, (222, 181), threshold=None)
mesh_display(heatmap_data, cmap=cmap)

save_directory = "output"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

plt.savefig("{}/{}".format(save_directory, "output"))