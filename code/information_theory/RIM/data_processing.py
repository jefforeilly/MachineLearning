#!/usr/bin/env python
# coding=utf-8
"""Data processing utilities.

This module allows you to:
1. Split datasets
2. Convert Pandas string column values to their discrete numerical counterparts
3. Get dataset; if file is not present it is downloaded from url

"""

import pathlib

import pandas as pd

__all__ = ['split_data', 'convert_to_numerical', 'get_dataset']


def get_dataset(dataset_path, dataset_url):
    """Retrieve dataset from file or url

    :param dataset_path: Path to fataset
    :param dataset_url: URL to dataset (downloaded if dataset_path does not
    exist)
    :returns: pandas DataFrame containing data
    """
    file_handle = pathlib.Path(dataset_path)
    if file_handle.is_file():
        try:
            return pd.read_csv(dataset_path, sep=',', header=None)
        except OSError:
            raise
    else:
        return pd.read_csv(dataset_url, sep=',', header=None)


def split_data(df, training=5, validation=1, testing=1, inplace=False):
    """Splits dataset into training, validation and testing w.r.t. given
    proportions.

    :param df: Pandas dataframe to split into training, testing and validation
    :param training: Proportion of testing dataset
    :param validation: Proportion of validation dataset
    :param testing: Proportion of testing dataset
    :param inplace: [Default: False] Shuffle pandas dataframe or make copy and
    shuffle
    :returns: List containg 3 Pandas Dataframes (training, validation, testing)

    """
    part = len(df) / (training + validation + testing)
    training = int(training * part)
    validation = training + int(validation * part)

    # COPY OR USE DATAFRAME
    dataframe = df if inplace else df.copy()
    # SHUFFLE DATA
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    return dataframe[:training], \
        dataframe[training:validation],\
        dataframe[validation:]


def convert_to_numerical(df, columns, inplace=False):
    """Converts values to their numerical categorical counterparts.

    :param df: unified pandas Dataframe object
    :param columns: df column's to be transformed
    :param inplace: perform transformation in-place or return dataframe
    (default behaviour, value=False)
    :returns: Dataset with converted values and their respective codes as
    dictionary (tuple)

    """
    dataframe = pd.DataFrame(df) if inplace else df
    codes = {}
    for column in columns:
        dataframe[column] = pd.Categorical(dataframe[column])
        codes[column] = dict(enumerate(dataframe[column].cat.categories))
        # WORKAROUND IF YOU WANT TO WORK WITH SPARSE MATRICES AFTERWARDS
        dataframe[column] = dataframe[column].cat.codes
    return dataframe, codes
