from random import randint

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

from read_data import slice_image


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
            arr.append(not_fire - fire)
        arr = np.array(arr)
        arr = (arr - arr.min()) / (arr.max() - arr.min())

    if threshold is not None:
        for i in range(len(arr)):
            if arr[i] > threshold:
                arr[i] = 1
            else:
                arr[i] = 0

    return np.reshape(arr, shape)



if __name__ == "__main__":
    arr = np.array([
        [0.3, 0.7],
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    heatmap = convert_output_to_heatmap(arr, (3, 1))
    print(heatmap)
    heatmap_display(heatmap)
    plt.show()
    '''
    import numpy as np
    arr = np.arange(1, 26)
    arr = arr.reshape((5, 5, 1))
    result = slice_image(arr, 5, 5)
    values = []
    for r in result:
        value = randint(0, 5)
        values.append(value)
    result = np.reshape(np.array(values), (5, 5))
    heatmap_display(result)
    plt.show()
    '''