import os

import matplotlib.pyplot as plt

from visualize import convert_output_to_heatmap, heatmap_display
from read_data import slice_image, get_data_from_folder, normalize_channels, softmax, get_folders
from config import data_abs, data

plt.axis('off')

def run_model(model, X):
    X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    outputs = model.predict(X)
    outputs = softmax(outputs, axis=1)
    return outputs


def main(model, X, y, name):
    model = model.fit(X, y)

    data_types = ["fire", "maybe_fire"]
    for data_type in data_types:
        folders = get_folders(data_abs[data_type])
        for folder in folders:
            folder_abs = "{}/{}".format(data_abs[data_type], folder)
            img_arr = get_data_from_folder(folder_abs)
            img_arr = normalize_channels(img_arr)
            img_arr = slice_image(img_arr, 5, 5)
            img_arr = np.nan_to_num(img_arr)

            outputs = run_model(model, img_arr)
            heatmap_data = convert_output_to_heatmap(outputs, (222, 181), threshold=None)
            heatmap_display(heatmap_data)

            save_directory = "{}/{}/{}".format(data['outputs'], data_type, name)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            #plt.show()
            plt.savefig("{}/{}".format(save_directory, folder))


if __name__ == "__main__":
    import numpy as np
    #from sklearn.linear_model import LinearRegression
    from ann import ArtificialNeuralNetwork

    X = np.load("data/features.npy")
    y = np.load("data/labels.npy")
    image_folder = "{}/{}".format(data_abs["fire"], "201805312300")

    X_train = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    X_train = np.nan_to_num(X_train)

    #model = LinearRegression(normalize=True)
    model = ArtificialNeuralNetwork(epochs=25)
    main(model, X_train, y, "neural_network")

