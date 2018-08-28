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


def main(model, X, y, name, thresholds=[None, 0.7, 0.8, 0.9, 0.99]):
    model = model.fit(X, y)

    data_types = ["fire"]
    for data_type in data_types:
        folders = get_folders(data_abs[data_type])
        for folder in folders:
            for threshold in thresholds:
                folder_abs = "{}/{}".format(data_abs[data_type], folder)
                img_arr = get_data_from_folder(folder_abs)
                img_arr = normalize_channels(img_arr)
                img_arr = slice_image(img_arr, 5, 5)
                img_arr = np.nan_to_num(img_arr)

                outputs = run_model(model, img_arr)
                if threshold is None:
                    cmap = "coolwarm"
                else:
                    cmap = "magma"

                heatmap_data = convert_output_to_heatmap(outputs, (222, 181), threshold=threshold)
                heatmap_display(heatmap_data, cmap=cmap)

                save_directory = "{}/{}/{}".format(data['outputs'], data_type, name)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                #plt.show()
                if threshold is not None:
                    plt.savefig("{}/{}-{}".format(save_directory, folder, int(threshold * 100)))
                else:
                    plt.savefig("{}/{}-{}".format(save_directory, folder, "none"))


def convert_to_binary(y):
    return np.argmax(y, axis=1)



if __name__ == "__main__":
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


    import sklearn

    X = np.load("data/features.npy")
    y = np.load("data/labels.npy")
    y_binary = convert_to_binary(y)

    image_folder = "{}/{}".format(data_abs["fire"], "201805312300")

    X_train = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
    X_train = np.nan_to_num(X_train)

    model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
    #model = sklearn.svm.SVC()
    #model = KNeighborsClassifier(n_neighbors=2)
    #model = GaussianNB()
    #model = MultinomialNB()
    #model = BernoulliNB()

    main(model, X_train, y, "LinearRegression")

