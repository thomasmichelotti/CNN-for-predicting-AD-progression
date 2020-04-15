# import configuration file
import config

# set random seed
#from numpy.random import seed
#from tensorflow import set_random_seed
#seed(config.fixed_seed)
#set_random_seed(config.fixed_seed)

import numpy as np
import keras
import h5py
import os

class DataGeneratorMultipleInputs(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, labels, non_image_data, mean, std, non_image_mean, non_image_std, batch_size=16, dim=(32,32,32), dim_non_image=(), n_channels=1,
                 n_classes=3, shuffle=True): # ADD MEAN, STD, DIM FOR NON-IMAGE-DATA AS WELL???
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs_temp = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        self.non_image_mean = non_image_mean
        self.non_image_std = non_image_std
        self.data_dir = config.data_dir
        self.aug_dir = config.aug_dir
        self.on_epoch_end()
        self.non_image_data = non_image_data
        self.dim_non_image = dim_non_image

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        # shuffle order of input examples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_image = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_non_image = np.empty((self.batch_size, *self.dim_non_image, )) #self.n_channels
        y = np.empty((self.batch_size), dtype=int)
        if config.ordered_softmax == True:
            Y = np.empty((self.batch_size, self.n_classes-1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample: load original and augmented images
            if ID[0] != 'a':
                if config.task == "CN-MCI-AD":
                    with h5py.File(self.data_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                    X_image[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X_image[i,] = np.reshape(np.load(self.data_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            else:
                if config.task == "CN-MCI-AD":
                    with h5py.File(self.aug_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                    X_image[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X_image[i,] = np.reshape(np.load(self.aug_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))

            # normalization
            X_image[i,] = np.subtract(X_image[i,], np.reshape(self.mean, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))
            X_image[i,] = np.divide(X_image[i,], np.reshape(self.std, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))

            # Store non-image data
            non_image_row = self.non_image_data[self.non_image_data["PTID_VISCODE"] == ID]
            #non_image_columns = non_image_row.loc[:, "MMSE":].to_numpy()
            non_image_columns = non_image_row[non_image_row.columns.intersection(config.feature_list)].to_numpy()
            X_non_image[i,] = non_image_columns[0]

            # normalization for non-imaging features
            X_non_image[i,] = np.subtract(X_non_image[i,], self.non_image_mean)
            X_non_image[i,] = np.divide(X_non_image[i,], self.non_image_std)

            # Store class
            y[i] = self.labels[ID]
            if config.ordered_softmax == True:
                if self.labels[ID] == 0:
                    Y[i,] = [0,0]
                elif self.labels[ID] == 1:
                    Y[i,] = [1,0]
                elif self.labels[ID] == 2:
                    Y[i,] = [1,1]

        X = [np.array(X_image), np.array(X_non_image)]
        if config.ordered_softmax == False:
            Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y





class DataGeneratorOnlyNonImagingFeatures(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, labels, non_image_data, non_image_mean, non_image_std, batch_size=16, dim_non_image=(), n_channels=1,
                 n_classes=3, shuffle=True): # ADD MEAN, STD, DIM FOR NON-IMAGE-DATA AS WELL???
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs_temp = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.non_image_mean = non_image_mean
        self.non_image_std = non_image_std
        self.data_dir = config.data_dir
        self.aug_dir = config.aug_dir
        self.on_epoch_end()
        self.non_image_data = non_image_data
        self.dim_non_image = dim_non_image

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        # shuffle order of input examples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_non_image = np.empty((self.batch_size, *self.dim_non_image, )) #self.n_channels
        y = np.empty((self.batch_size), dtype=int)
        if config.ordered_softmax == True:
            Y = np.empty((self.batch_size, self.n_classes-1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store non-image data
            non_image_row = self.non_image_data[self.non_image_data["PTID_VISCODE"] == ID]
            #non_image_columns = non_image_row.loc[:, "MMSE":].to_numpy()
            non_image_columns = non_image_row[non_image_row.columns.intersection(config.feature_list)].to_numpy()
            X_non_image[i,] = non_image_columns[0]

            # normalization for non-imaging features
            X_non_image[i,] = np.subtract(X_non_image[i,], self.non_image_mean)
            X_non_image[i,] = np.divide(X_non_image[i,], self.non_image_std)

            # Store class
            y[i] = self.labels[ID]
            if config.ordered_softmax == True:
                if self.labels[ID] == 0:
                    Y[i,] = [0,0]
                elif self.labels[ID] == 1:
                    Y[i,] = [1,0]
                elif self.labels[ID] == 2:
                    Y[i,] = [1,1]

        X = np.array(X_non_image)
        if config.ordered_softmax == False:
            Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y



class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, labels, mean, std, batch_size=16, dim=(32,32,32), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs_temp = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        self.data_dir = config.data_dir
        self.aug_dir = config.aug_dir
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        # shuffle order of input examples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        if config.ordered_softmax == True:
            Y = np.empty((self.batch_size, self.n_classes-1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample: load original and augmented images
            if ID[0] != 'a':
                if config.task == "CN-MCI-AD":
                    with h5py.File(self.data_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                    X[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X[i,] = np.reshape(np.load(self.data_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            else:
                if config.task == "CN-MCI-AD":
                    with h5py.File(self.aug_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                    X[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X[i,] = np.reshape(np.load(self.aug_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))

            # normalization
            X[i,] = np.subtract(X[i,], np.reshape(self.mean, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))
            X[i,] = np.divide(X[i,], np.reshape(self.std, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))

            # Store class
            y[i] = self.labels[ID]
            if config.ordered_softmax == True:
                if self.labels[ID] == 0:
                    Y[i,] = [0,0]
                elif self.labels[ID] == 1:
                    Y[i,] = [1,0]
                elif self.labels[ID] == 2:
                    Y[i,] = [1,1]

        if config.ordered_softmax == False:
            Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Below are the same three generators as above (only images, only non-imaging features, and both images and non-imaging features.
# The difference is that in the generators below, the unique scans per individual in a given interval that were chosen can randomly be replace by another scan of the same individual in the same interval
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class DataGeneratorMultipleInputs_random_scans(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, list_IDs_link, labels, df, non_image_data, mean, std, non_image_mean, non_image_std, batch_size=16, dim=(32,32,32), dim_non_image=(), n_channels=1,
                 n_classes=3, shuffle=True): # ADD MEAN, STD, DIM FOR NON-IMAGE-DATA AS WELL???
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs_link = list_IDs_link
        self.list_IDs_temp = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        self.non_image_mean = non_image_mean
        self.non_image_std = non_image_std
        self.data_dir = config.data_dir
        self.aug_dir = config.aug_dir
        self.on_epoch_end()
        self.non_image_data = non_image_data
        self.dim_non_image = dim_non_image
        self.df = df

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        #print(self.shuffle, flush=True)
        # shuffle order of input examples
        if self.shuffle == True:
            #print("I'm going to shuffle!", flush=True)
            np.random.shuffle(self.indexes)
        #print(self.indexes, flush= True)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_image = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_non_image = np.empty((self.batch_size, *self.dim_non_image, )) #self.n_channels
        # if config.extra_penalty_misclassified_converters == False:
        #     y = np.empty((self.batch_size), dtype=int)
        # else:
        #     y = np.empty((self.batch_size, 2), dtype=int)
        y = np.empty((self.batch_size), dtype=int)
        if config.ordered_softmax == True:
            Y = np.empty((self.batch_size, self.n_classes-1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample: load original and augmented images
            if ID[0] != 'a':
                if config.task == "CN-MCI-AD":
                    if config.TADPOLE_setup == False:
                        # choose random available scan for this individual
                        RID = self.df.loc[self.df["PTID_VISCODE"] == ID, "RID"].iloc[0]
                        temp_DF = self.df[(self.df["RID"] == RID) & (self.df["Diagnosis"] == self.labels[ID] + 1)]
                        unique_DF = temp_DF.drop_duplicates("PTID_VISCODE", keep="last")
                        available_scans = unique_DF["PTID_VISCODE"].copy()
                        new_ID = np.random.choice(available_scans)
                    else:
                        new_ID = ID
                    with h5py.File(self.data_dir + new_ID + '.h5py', 'r') as hf:
                        x = hf[new_ID][:]
                    X_image[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X_image[i,] = np.reshape(np.load(self.data_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            else:
                new_ID = ID
                if config.task == "CN-MCI-AD":
                    with h5py.File(self.aug_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                    X_image[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X_image[i,] = np.reshape(np.load(self.aug_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))

            # normalization
            X_image[i,] = np.subtract(X_image[i,], np.reshape(self.mean, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))
            X_image[i,] = np.divide(X_image[i,], np.reshape(self.std, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))

            # Store non-image data
            non_image_row = self.non_image_data[self.non_image_data["PTID_VISCODE"] == new_ID]
            unique_row = non_image_row.drop_duplicates("PTID_VISCODE", keep="last")
            #non_image_columns = non_image_row.loc[:, "MMSE":].to_numpy()
            non_image_columns = unique_row[unique_row.columns.intersection(config.feature_list)].to_numpy()
            X_non_image[i,] = non_image_columns[0]

            # temp check
            if (X_non_image[i,0] == 1) and (self.labels[new_ID] == 1):
                checkpoint = 1

            # normalization for non-imaging features
            X_non_image[i,] = np.subtract(X_non_image[i,], self.non_image_mean)
            X_non_image[i,] = np.divide(X_non_image[i,], self.non_image_std)

            # Store class
            y[i] = self.labels[new_ID]
            # y[i,1] = self.labels[ID] # labels[ID] is the same as labels[new_ID]
            # index = non_image_row.index.item()
            # y[i,2] = non_image_row[index, "Current_diagnosis"]
            if config.ordered_softmax == True:
                if self.labels[ID] == 0:
                    Y[i,] = [0,0]
                elif self.labels[ID] == 1:
                    Y[i,] = [1,0]
                elif self.labels[ID] == 2:
                    Y[i,] = [1,1]

        X = [np.array(X_image), np.array(X_non_image)]
        # if config.ordered_softmax == False:
        #     if config.extra_penalty_misclassified_converters == False:
        #         Y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        #     else:
        #         Y = np.concatenate((keras.utils.to_categorical(y[:,1], num_classes=self.n_classes), y[:,2]), axis=0)
        if config.ordered_softmax == False:
            Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y





class DataGeneratorOnlyNonImagingFeatures_random_scans(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, list_IDs_link, labels, df, non_image_data, non_image_mean, non_image_std, batch_size=16, dim_non_image=(), n_channels=1,
                 n_classes=3, shuffle=True): # ADD MEAN, STD, DIM FOR NON-IMAGE-DATA AS WELL???
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.list_IDs_link = list_IDs_link
        self.list_IDs_temp = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.non_image_mean = non_image_mean
        self.non_image_std = non_image_std
        self.data_dir = config.data_dir
        self.aug_dir = config.aug_dir
        self.on_epoch_end()
        self.non_image_data = non_image_data
        self.dim_non_image = dim_non_image
        self.df = df

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        # shuffle order of input examples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_non_image = np.empty((self.batch_size, *self.dim_non_image, )) #self.n_channels
        y = np.empty((self.batch_size), dtype=int)
        if config.ordered_softmax == True:
            Y = np.empty((self.batch_size, self.n_classes-1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store non-image data
            non_image_row = self.non_image_data[self.non_image_data["PTID_VISCODE"] == ID]
            #non_image_columns = non_image_row.loc[:, "MMSE":].to_numpy()
            non_image_columns = non_image_row[non_image_row.columns.intersection(config.feature_list)].to_numpy()
            X_non_image[i,] = non_image_columns[0]

            # normalization for non-imaging features
            X_non_image[i,] = np.subtract(X_non_image[i,], self.non_image_mean)
            X_non_image[i,] = np.divide(X_non_image[i,], self.non_image_std)

            # Store class
            y[i] = self.labels[ID]
            if config.ordered_softmax == True:
                if self.labels[ID] == 0:
                    Y[i,] = [0,0]
                elif self.labels[ID] == 1:
                    Y[i,] = [1,0]
                elif self.labels[ID] == 2:
                    Y[i,] = [1,1]

        X = np.array(X_non_image)
        if config.ordered_softmax == False:
            Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y



class DataGenerator_random_scans(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, list_IDs_link, labels, df, mean, std, batch_size=16, dim=(32,32,32), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        #self.labels_link = labels_link
        self.list_IDs = list_IDs
        self.list_IDs_link = list_IDs_link
        self.list_IDs_temp = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.mean = mean
        self.std = std
        self.data_dir = config.data_dir
        self.aug_dir = config.aug_dir
        self.df = df
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

        # shuffle order of input examples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        if config.ordered_softmax == True:
            Y = np.empty((self.batch_size, self.n_classes-1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample: load original and augmented images
            if ID[0] != 'a':
                if config.task == "CN-MCI-AD":
                    if config.TADPOLE_setup == False:
                        # choose random available scan for this individual
                        RID = self.df.loc[self.df["PTID_VISCODE"] == ID, "RID"].iloc[0]
                        temp_DF = self.df[(self.df["RID"] == RID) & (self.df["Diagnosis"] == self.labels[ID]+1)]
                        unique_DF = temp_DF.drop_duplicates("PTID_VISCODE", keep="last")
                        available_scans = unique_DF["PTID_VISCODE"].copy()
                        new_ID = np.random.choice(available_scans)
                    else:
                        new_ID = ID

                    with h5py.File(self.data_dir + new_ID + '.h5py', 'r') as hf:
                        x = hf[new_ID][:]
                    X[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X[i,] = np.reshape(np.load(self.data_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            else:
                new_ID = ID
                if config.task == "CN-MCI-AD":
                    with h5py.File(self.aug_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                    X[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
                else:
                    X[i,] = np.reshape(np.load(self.aug_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))

            # normalization
            X[i,] = np.subtract(X[i,], np.reshape(self.mean, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))
            X[i,] = np.divide(X[i,], np.reshape(self.std, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))

            # Store class
            y[i] = self.labels[new_ID]
            if config.ordered_softmax == True:
                if self.labels[ID] == 0:
                    Y[i,] = [0,0]
                elif self.labels[ID] == 1:
                    Y[i,] = [1,0]
                elif self.labels[ID] == 2:
                    Y[i,] = [1,1]

        if config.ordered_softmax == False:
            Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y


