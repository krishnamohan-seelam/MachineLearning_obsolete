import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import matplotlib


def auto_scatter_simple(df, plot_cols, target_cols, filename):
    matplotlib.use('agg')
    for target_col in target_cols:

        for col in plot_cols:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.gca()
            ## simple scatter plot
            df.plot(kind='scatter', x=col, y=target_col, ax=ax, color='DarkBlue')
            ax.set_title('Scatter plot of {0} vs. {1}'.format(target_col, col))
            outputfile = getoutFileName(filename, 'png')
            print(outputfile)
            fig.savefig(outputfile)
        return plot_cols


def getoutFileName(inputfile, target_col, extension):
    fileName, fileExtension = os.path.splitext(inputfile)
    print(fileName, fileExtension)
    extension = "." + extension

    if not fileExtension:
        return fileName + target_col + extension

    if not (fileExtension and not fileExtension.isspace()):
        return fileName + target_col + extension

    if not inputfile.endswith(extension):
        return inputfile.replace(fileExtension, target_col + extension)


def load_data(filename, separator=","):
    data_set = None
    if not os.path.exists(filename):
        print("{0} file not found".format(filename))
        return None

    if filename:
        try:
            data_set = pd.read_csv(filename, sep=separator, header=0, encoding='utf8')
        except FileNotFoundError:
            print("{0} file not found".format(filename))
            return None
    return data_set


def print_dataset_info(dataset):
    if dataset is None:
        print("data set is EMPTY")
    else:
        print("No of Observation:{0}".format(dataset.shape[0]))
        print("No of features:{0}".format(dataset.shape[1]))
        print("Features:{0}".format(dataset.columns.values))
        print("Describe dataset:{0}".format(dataset.describe()))


def split_dataset(train_X, train_Y, ptest_size=0.3, prandom_state=1):
    (train_data, test_data, train_target, test_target) = train_test_split(train_X, train_Y,
                                                                          test_size=ptest_size,
                                                                          random_state=prandom_state)
    return (train_data, test_data, train_target, test_target)


def print_compare_results(test_target, prediction_dataset):
    test_values = test_target.values.reshape(-1, 1)
    for i, prediction in enumerate(prediction_dataset.reshape(-1, 1)):
        print('Predicted: ', prediction)
        print('Target: ', test_values[i])
