# import configuration file
import config
import h5py
import os

# set random seed
#from numpy.random import seed
#from tensorflow import set_random_seed
#seed(config.fixed_seed)
#set_random_seed(config.fixed_seed)

import numpy as np


def standardization_matrix(dataset):
    """
    Returns the voxel-wise mean and std of an input dataset
    Note: the standardization matrix computed from the training data should also be
    used for the validation and test data.

        INPUT:
            dataset - the dataset containing all IDs of the subjects which should be used for normalization

        OUTPUT:
            mean - 3D matrix containing the voxel-wise mean computed from all subjects of the input dataset
            std - 3D matrix containing the voxel-wise std computed from all subjects of the input dataset
    """

    print("\nCompute standardization matrix")
    mean = mean_matrix(dataset)
    print("    mean : computed")
    std = std_matrix(mean, dataset)
    print("    std : computed\n")

    return mean, std


def mean_matrix(dataset):
    """
    Returns the voxel-wise mean of an input dataset
    """

    print("    mean : in progress")

    # loop over every input file and calculate mean
    mean = np.zeros(config.input_shape)
    for id in dataset:
        if id[0] != 'a':
            if config.task == "CN-MCI-AD":
                with h5py.File(config.data_dir + id + '.h5py', 'r') as hf:
                    x = hf[id][:]
            else:
                x = np.load(config.data_dir + id + '.npy')
        else:
            if config.task == "CN-MCI-AD":
                with h5py.File(config.aug_dir + id + '.h5py', 'r') as hf:
                    x = hf[id][:]
            else:
                x = np.load(config.aug_dir + id + '.npy') # change if data augmentation is used
        mean = mean + x / len(dataset)

    return mean


def std_matrix(mean, dataset):
    """
    Returns the voxel-wise std of an input dataset based on the mean of that dataset
    """

    print("    std : in progress")

    # loop over every input file and calculate std
    std = np.zeros(config.input_shape)
    for id in dataset:
        if id[0] != 'a':
            if config.task == "CN-MCI-AD":
                with h5py.File(config.data_dir + id + '.h5py', 'r') as hf:
                    x = hf[id][:]
            else:
                x = np.load(config.data_dir + id + '.npy')
        else:
            if config.task == "CN-MCI-AD":
                with h5py.File(config.aug_dir + id + '.h5py', 'r') as hf:
                    x = hf[id][:]
            else:
                x = np.load(config.aug_dir + id + '.npy') # change if data augmentation is used
        std = std + np.square(x - mean) / len(dataset)

    std = np.sqrt(std)

    # to avoid dividing by 0 add 1e-20 to the std
    std = std + np.ones(config.input_shape) * 1e-20

    return std


# NON-IMAGING STANDARDIZATION

def non_image_standardization_matrix(features):
    """
    Returns the voxel-wise mean and std of an input dataset
    Note: the standardization matrix computed from the training data should also be
    used for the validation and test data.

        INPUT:
            dataset - the dataset containing all IDs of the subjects which should be used for normalization # CHANGE

        OUTPUT:
            mean - 3D matrix containing the voxel-wise mean computed from all subjects of the input dataset
            std - 3D matrix containing the voxel-wise std computed from all subjects of the input dataset
    """

    print("\nCompute standardization matrix")
    non_image_mean = non_image_mean_matrix(features)
    print("    non-image mean : computed")
    non_image_std = non_image_std_matrix(non_image_mean, features)
    print("    non-image std : computed\n")

    return non_image_mean, non_image_std


def non_image_mean_matrix(features):
    """
    Returns the voxel-wise mean of an input dataset
    """

    print("    non-image mean : in progress")

    # loop over every input file and calculate mean
    #non_image_mean = np.zeros(config.non_image_input_shape)
    #features = features.loc[:, "MMSE":].to_numpy()
    features = features[features.columns.intersection(config.feature_list)].to_numpy()
    non_image_mean = np.mean(features, axis=0)


    # for id in dataset:
    #     if id[0] != 'a':
    #         if config.task == "CN-MCI-AD":
    #             # with h5py.File(config.data_dir + id + '.h5py', 'r') as hf:
    #             #     x = hf[id][:]
    #             non_image_row = features[features["PTID_VISCODE"] == id]
    #             non_image_columns = non_image_row.loc[:, "MMSE":].to_numpy()
    #             X_non_image[i,] = non_image_columns[0]
    #         else:
    #             x = np.load(config.data_dir + id + '.npy') # DOES NOT WORK FOR STANDARDIZATION OF NON-IMAGING FEATURES
    #     else:
    #         x = np.load(config.aug_dir + id + '.npy') # change if data augmentation is used, DOES NOT WORK FOR STANDARDIZATION OF NON-IMAGING FEATURES
    #     non_image_mean = non_image_mean + x / len(dataset)

    return non_image_mean


def non_image_std_matrix(non_image_mean, features):
    """
    Returns the voxel-wise std of an input dataset based on the mean of that dataset
    """

    print("    non-image std : in progress")

    # loop over every input file and calculate std
    non_image_std = np.zeros(config.non_image_input_shape)
    features = features[features.columns.intersection(config.feature_list)].to_numpy()
    #features = features.loc[:, "MMSE":].to_numpy()

    for row in range(features.shape[0]):
        non_image_std = non_image_std + np.square(features[row] - non_image_mean) / features.shape[0]


    # for id in dataset:
    #     if id[0] != 'a':
    #         if config.task == "CN-MCI-AD":
    #             with h5py.File(config.data_dir + id + '.h5py', 'r') as hf:
    #                 x = hf[id][:]
    #         else:
    #             x = np.load(config.data_dir + id + '.npy')
    #     else:
    #         x = np.load(config.aug_dir + id + '.npy') # change if data augmentation is used
    #     non_image_std = non_image_std + np.square(x - non_image_mean) / len(dataset)
    #
    # non_image_std = np.sqrt(non_image_std)

    # to avoid dividing by 0 add 1e-20 to the std
    non_image_std = non_image_std + np.ones(config.non_image_input_shape) * 1e-20

    return non_image_std