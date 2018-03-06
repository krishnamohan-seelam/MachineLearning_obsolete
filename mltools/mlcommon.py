import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np

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
        raise ValueError("{0} not found".format(filename))

    if filename:
        try:
            data_set = pd.read_csv(filename, sep=separator, header=0, encoding='utf8')
        except FileNotFoundError:
            print("{0} file not found".format(filename))
            return None
    return data_set

def load_dataset(input_file,response,colseparator=','):
    
    input_dataset = load_data(input_file,colseparator)
    print(" input file is :{0} loaded.".format(input_file))
    #print(input_dataset.head())
    
    try:
        continuous_vars = input_dataset.describe().columns.values.tolist()
        print("Continous Variables")
        print(continuous_vars)
    except ValueError:
        print("No continous variables")
    
    try:
        categorical_vars = input_dataset.describe(include=["object"]).columns.values.tolist()
        print("Categorical Variables")
        print(categorical_vars)
    except ValueError:
        print("No categorical variables")
        categorical_vars = None
    
    response_column =  [col for col in input_dataset.columns if response == col]
    feature_columns =  [col for col in input_dataset.columns if response != col]
      
    return  input_dataset,feature_columns,response_column,continuous_vars,categorical_vars


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



def detect_outliers(dataset,noutliers,columns):
    outlier_indices = []
    for column in columns:
        # 1st quartile (25%),# 3rd quartile (75%)
        q1, q3 = np.percentile(dataset[column], [25, 75])
         
       # Interquartile range (IQR)
        iqr = q3 - q1
        
        # outlier step
        outlier_step = 1.5 * iqr
        
        lower_bound = q1 - outlier_step
        upper_bound = q3 + outlier_step
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = dataset[(dataset[column] < lower_bound ) | (dataset[column] > upper_bound )].index
        outlier_indices.extend(outlier_list_col)
         
    outlier_indices = Counter(outlier_indices)
     
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > noutliers )
    
    return multiple_outliers 

def display_data_descriptives(input_dataset,feature_columns,response_column):
    print("<{0} {1} {0}>".format("="*40,"info"))
    print(input_dataset.info())
    print("<{0} {1} {0}>".format("="*40,"feature columns"))
    print(feature_columns)
    print("<{0} {1} {0}>".format("="*40,"response"))
    print(response_column)
    print("<{0} {1} {0}>".format("="*40,"Descriptive Statistics -X"))
    print(input_dataset[feature_columns].describe())
    print("<{0} {1} {0}>".format("="*40,"Descriptive Statistics -y"))
    print(input_dataset[response_column].describe())

    
def print_compare_results(test_target, prediction_dataset):
    test_values = test_target.values.reshape(-1, 1)
    for i, prediction in enumerate(prediction_dataset.reshape(-1, 1)):
        print('Predicted: ', prediction)
        print('Target: ', test_values[i])
