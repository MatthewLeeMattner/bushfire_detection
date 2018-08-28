from random import randint

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from read_data import slice_image


def heatmap_display(outputs):
    fig, ax = plt.subplots(figsize=(10, 12.25))
    ax.set_yticks([])
    ax.set_xticks([])
    sns.heatmap(outputs, cmap="coolwarm", ax=ax)


def convert_output_to_heatmap(outputs, shape, threshold=None):
    arr = []
    for fire, not_fire in outputs:
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