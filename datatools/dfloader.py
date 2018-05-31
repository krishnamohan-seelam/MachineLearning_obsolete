"""
Inspired from Randal S. Olson's autoclean.
Objective of this class to split dataframe into categorical & continuous frames. 
"""
from __future__ import print_function
import numpy as np
import pandas as pd

from os.path import splitext
VALID_FORMATS = ["csv"]


class DataFrameLoader():

    def __init__(self, *args, **kwargs):
        self.__dataframe = self.load_data(*args, **kwargs)
        self.__categorical_features = []
        self.__continuous_features = []
        if not self.__dataframe.empty:
            self.__categorical_features = self.__get_features_by_type(
                type="object")
            self.__continuous_features = self.__get_features_by_type(
                type=np.number)

    @property
    def dataframe(self):
        return self.__dataframe

    @property
    def categorical_features(self):
        return self.__categorical_features

    @property
    def continuous_features(self):
        return self.__continuous_features

    @property
    def numerical_dataframe(self):
        return self.__dataframe[self.continuous_features]

    @property
    def categorical_dataframe(self):
        return self.__dataframe[self.categorical_features]

    def load_data(self, *args, **kwargs):
        _, extension = splitext(args[0])
        if extension[1:].lower() == 'csv':
            return self.load_csv(*args, **kwargs)

    def load_csv(self, *args, **kwargs):
        dataframe = pd.DataFrame()
        try:
            dataframe = pd.read_csv(*args, **kwargs)
             
        except Exception:
            print("Error in loading data, empty data frame returned")

        return dataframe

    def __get_features_by_type(self, type):
        features = self.__dataframe.describe(include=[type]).columns.values
        features = features if features.any() else []
        return features


def main():
    loader = DataFrameLoader('train.csv', sep=',')
    dataframe = loader.dataframe
    print(dataframe.head())
    print(loader.categorical_features)
    print(loader.continuous_features)
    print(loader.categorical_dataframe.head())
    print(loader.numerical_dataframe.head())


if __name__ == '__main__':
    main()
