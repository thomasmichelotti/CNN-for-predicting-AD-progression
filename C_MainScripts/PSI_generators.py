import numpy as np
import keras
import h5py
import config_PSI
import os

class PSI_DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """
    def __init__(self, list_IDs, labels, mean, std, batch_size=1, dim=(32,32,32), n_channels=1,
                 n_classes=3, shuffle=False):
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
        self.data_dir = config_PSI.image_dir
        #self.aug_dir = config.aug_dir
        self.on_epoch_end()
        self.count = 0

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
        #if config.ordered_softmax == True:
        #    Y = np.empty((self.batch_size, self.n_classes-1))

        self.count += 1
        print(str(self.count) + " / " + str(len(self.list_IDs)))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample: load original and augmented images
            # if ID[0] != 'a':
            if config_PSI.task == "CN-MCI-AD":
                if os.path.exists(self.data_dir + ID + '.h5py'):
                    with h5py.File(self.data_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                else:
                    if ID[3] == "_":
                        temp_ID = ID[0:3] + ID[4:]
                        with h5py.File(self.data_dir + temp_ID + '.h5py', 'r') as hf:
                            x = hf[temp_ID][:]
                    else:
                        temp_ID = ID[0:3] + "_" + ID[3:]
                        with h5py.File(self.data_dir + temp_ID + '.h5py', 'r') as hf:
                            x = hf[temp_ID][:]
                X[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            else:
                X[i,] = np.reshape(np.load(self.data_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            # else:
            #     if config.task == "CN-MCI-AD":
            #         with h5py.File(self.aug_dir + ID + '.h5py', 'r') as hf:
            #             x = hf[ID][:]
            #         X[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            #     else:
            #         X[i,] = np.reshape(np.load(self.aug_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))

            # normalization
            X[i,] = np.subtract(X[i,], np.reshape(self.mean, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))
            X[i,] = np.divide(X[i,], np.reshape(self.std, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))

            # Store class
            y[i] = self.labels[ID]
            # if config.ordered_softmax == True:
            #     if self.labels[ID] == 0:
            #         Y[i,] = [0,0]
            #     elif self.labels[ID] == 1:
            #         Y[i,] = [1,0]
            #     elif self.labels[ID] == 2:
            #         Y[i,] = [1,1]

        # if config.ordered_softmax == False:
        Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y





class PSI_DataGenerator_multiple_inputs(keras.utils.Sequence):
    """
    Generates data for Keras
    Code is adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    All images are loaded and normalized based on the given mean and std of the train set.
    """

    def __init__(self, list_IDs, labels, non_image_data, mean, std, non_image_mean, non_image_std, batch_size=16,
                 dim=(32, 32, 32), dim_non_image=(), n_channels=1,
                 n_classes=3, shuffle=False):
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
        self.data_dir = config_PSI.image_dir
        #self.aug_dir = config.aug_dir
        self.on_epoch_end()
        self.non_image_data = non_image_data
        self.dim_non_image = dim_non_image
        self.count = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        # samples / batch size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

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
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_image = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_non_image = np.empty((self.batch_size, *self.dim_non_image,))  # self.n_channels
        y = np.empty((self.batch_size), dtype=int)
        # if config.ordered_softmax == True:
        #     Y = np.empty((self.batch_size, self.n_classes - 1))

        self.count += 1
        print(str(self.count) + " / " + str(len(self.list_IDs)))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample: load original and augmented images
            # if ID[0] != 'a':
            if config_PSI.task == "CN-MCI-AD":
                if os.path.exists(self.data_dir + ID + '.h5py'):
                    with h5py.File(self.data_dir + ID + '.h5py', 'r') as hf:
                        x = hf[ID][:]
                else:
                    if ID[3] == "_":
                        temp_ID = ID[0:3] + ID[4:]
                        with h5py.File(self.data_dir + temp_ID + '.h5py', 'r') as hf:
                            x = hf[temp_ID][:]
                    else:
                        temp_ID = ID[0:3] + "_" + ID[3:]
                        with h5py.File(self.data_dir + temp_ID + '.h5py', 'r') as hf:
                            x = hf[temp_ID][:]
                X_image[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            else:
                X_image[i,] = np.reshape(np.load(self.data_dir + ID + '.npy'), (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            # else:
            #     if config.task == "CN-MCI-AD":
            #         with h5py.File(self.aug_dir + ID + '.h5py', 'r') as hf:
            #             x = hf[ID][:]
            #         X_image[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
            #     else:
            #         X_image[i,] = np.reshape(np.load(self.aug_dir + ID + '.npy'),
            #                                  (self.dim[0], self.dim[1], self.dim[2], self.n_channels))

            # normalization
            X_image[i,] = np.subtract(X_image[i,], np.reshape(self.mean, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))
            X_image[i,] = np.divide(X_image[i,], np.reshape(self.std, (self.dim[0], self.dim[1], self.dim[2], self.n_channels)))

            # Store non-image data
            non_image_row = self.non_image_data[self.non_image_data["RID"] == ID]                                      # was PTID_VISCODE
            # non_image_columns = non_image_row.loc[:, "MMSE":].to_numpy()
            non_image_columns = non_image_row[non_image_row.columns.intersection(config_PSI.feature_list)].to_numpy()
            X_non_image[i,] = non_image_columns[0]

            # normalization for non-imaging features
            X_non_image[i,] = np.subtract(X_non_image[i,], self.non_image_mean)
            X_non_image[i,] = np.divide(X_non_image[i,], self.non_image_std)

            # Store class
            y[i] = self.labels[ID]
            # if config.ordered_softmax == True:
            #     if self.labels[ID] == 0:
            #         Y[i,] = [0, 0]
            #     elif self.labels[ID] == 1:
            #         Y[i,] = [1, 0]
            #     elif self.labels[ID] == 2:
            #         Y[i,] = [1, 1]

        X = [np.array(X_image), np.array(X_non_image)]
        #if config.ordered_softmax == False:
        Y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, Y