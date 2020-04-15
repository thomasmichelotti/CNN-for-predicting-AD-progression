# import configuration file
import config

import keras
from keras import Input, Model
from keras.layers import Conv3D, BatchNormalization, Activation, Dropout, GlobalAveragePooling3D, Dense
import C_MainScripts.ordinal_categorical_crossentropy as OCC
from keras.engine.saving import load_model

def build_model_allCNN(non_image_mean, non_image_std):
    """
    Builds a 3D all-CNN model which can be used for AD classification based on MRI.

        OUTPUT:
            model - the Keras implementation of the all-CNN
    """

    # INPUT
    input_image = Input(shape=(config.input_shape[0], config.input_shape[1], config.input_shape[2], 1))
    if config.non_image_data == True:
        non_image_input = Input(shape=(config.non_image_input_shape, ))

    if config.only_non_imaging == False:
        # use smaller model for down sampled data
        if config.roi == "GM_f4":

            if config.test_broader_cnn == False:
                # BLOCK 1
                x = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(input_image)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 2
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 3
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            else:
                # BLOCK 1
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(input_image)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 2
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 3
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            if config.test_deeper_cnn == True:
                # BLOCK 4
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

        # use more blocks and kernels for whole brain data
        else:

            # BLOCK 1
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(input_image)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 2
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 3
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 4
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 5
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 6
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        # BLOCK 7: 1-by-1 convolutions
        x = Conv3D(filters=16, kernel_size=(1, 1, 1), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # LAST: 1-by-1 convolution + kernel size of 3
        # if config.task == "AD" or config.task == "MCI":
        #     x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
        # elif config.task == "CN-MCI-AD":
        #     if config.ordered_softmax == False:
        #         x = Conv3D(filters=3, kernel_size=(1, 1, 1), padding='valid')(x)
        #     else:
        #         x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
        # x = Dropout(config.dropout)(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        if config.non_image_data == False:
            # LAST: 1-by-1 convolution + kernel size of 3
            if config.task == "AD" or config.task == "MCI":
                x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
            elif config.task == "CN-MCI-AD":
                if config.ordered_softmax == False:
                    x = Conv3D(filters=3, kernel_size=(1, 1, 1), padding='valid')(x)
                else:
                    x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = GlobalAveragePooling3D()(x)

            # OUTPUT
            if config.ordered_softmax == False:
                predictions = Activation('softmax')(x)
            else:
                predictions = Activation('sigmoid')(x)
            model = Model(inputs=input_image, outputs=predictions)
        elif config.non_image_data == True:
            x = GlobalAveragePooling3D()(x)
            x = keras.layers.concatenate([x, non_image_input])

            # x = Dense(32, activation='relu')(x) # new
            # x = Dense(32, activation='relu')(x)
            # x = Dense(16, activation='relu')(x) # new
            # x = Dense(16, activation='relu')(x)
            # x = Dense(8, activation='relu')(x) # new
            # x = Dense(8, activation='relu')(x)  # new
            # x = Dense(3, activation='relu')(x)

            #x = Dense(16, activation='relu')(x)
            #x = Dense(32, activation='relu')(x)
            x = Dense(16, activation='relu')(x)
            #x = Dense(8, activation='relu')(x)
            x = Dense(3, activation='relu')(x)

            # OUTPUT
            if config.ordered_softmax == False:
                mixed_output = Activation('softmax')(x)
            else:
                mixed_output = Activation('sigmoid')(x)
            model = Model(inputs=[input_image, non_image_input], outputs=mixed_output)

        if config.task == "AD" or config.task == "MCI":
            model.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                        metrics=['accuracy'])
        elif config.task == "CN-MCI-AD":
            if config.lr_optimiser == "Adam":
                if config.loss_function == OCC.customLossFunction:
                    model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std),
                                  optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                                  metrics=['categorical_accuracy'])
                else:
                    model.compile(loss=config.loss_function,
                                  optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                                  metrics=['categorical_accuracy'])
            elif config.lr_optimiser == "Adadelta":
                if config.loss_function == OCC.customLossFunction:
                    model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std), optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                                  metrics=['categorical_accuracy'])
                else:
                    model.compile(loss=config.loss_function, optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                                  metrics=['categorical_accuracy'])

            # keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False), keras.optimizers.Adadelta(lr=1.0, rho=0.95)
            # categorical_crossentropy, categorical_hinge

    else: # if config.only_non_imaging == True
        # x = Dense(32, activation='relu')(non_image_input)  # new
        # # x = BatchNormalization()(x)
        # x = Dense(32, activation='relu')(x)
        # # x = BatchNormalization()(x)
        # x = Dense(16, activation='relu')(x)  # new
        # # x = BatchNormalization()(x)
        # x = Dense(16, activation='relu')(x)
        # # x = BatchNormalization()(x)
        # x = Dense(8, activation='relu')(x)  # new
        # # x = BatchNormalization()(x)
        # x = Dense(8, activation='relu')(x)  # new
        # # x = BatchNormalization()(x)
        # x = Dense(3, activation='relu')(x)

        # x = Dense(16, activation='relu')(x)
        x = Dense(16, activation='relu')(non_image_input)
        x = Dense(3, activation='relu')(x)

        # OUTPUT
        if config.ordered_softmax == False:
            output = Activation('softmax')(x)
        else:
            output = Activation('sigmoid')(x)
        model = Model(inputs=non_image_input, outputs=output)

        if config.lr_optimiser == "Adam":
            if config.loss_function == OCC.customLossFunction:
                model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std),
                              optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                              metrics=['categorical_accuracy'])
            else:
                model.compile(loss=config.loss_function,
                              optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                              metrics=['categorical_accuracy'])
        elif config.lr_optimiser == "Adadelta":
            if config.loss_function == OCC.customLossFunction:
                model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std), optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                              metrics=['categorical_accuracy'])
            else:
                model.compile(loss=config.loss_function, optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                              metrics=['categorical_accuracy'])

    return model







def build_pretrained_model_allCNN(pretrained_model, non_image_mean, non_image_std):
    """
    Builds a pretrained 3D all-CNN model which can be used for AD classification based on MRI.

        OUTPUT:
            model - the Keras implementation of the pretrained all-CNN
    """

    # INPUT
    input_image = Input(shape=(config.input_shape[0], config.input_shape[1], config.input_shape[2], 1))
    if config.non_image_data == True:
        non_image_input = Input(shape=(config.non_image_input_shape, ))

    if config.only_non_imaging == False:
        # use smaller model for down sampled data
        if config.roi == "GM_f4":

            if config.test_broader_cnn == False:
                # BLOCK 1
                x = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(input_image)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 2
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 3
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            else:
                # BLOCK 1
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(input_image)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 2
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=24, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 3
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

            if config.test_deeper_cnn == True:
                # BLOCK 4
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # BLOCK 4
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

                # pooling
                x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',
                           kernel_regularizer=config.weight_regularize)(x)
                x = Dropout(config.dropout)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

        # use more blocks and kernels for whole brain data
        else:

            # BLOCK 1
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(input_image)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 2
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 3
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 4
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 5
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # BLOCK 6
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # pooling
            x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        # BLOCK 7: 1-by-1 convolutions
        x = Conv3D(filters=16, kernel_size=(1, 1, 1), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # pooling
        x = Conv3D(filters=16, kernel_size=(1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=config.weight_regularize)(x)
        x = Dropout(config.dropout)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # LAST: 1-by-1 convolution + kernel size of 3
        # if config.task == "AD" or config.task == "MCI":
        #     x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
        # elif config.task == "CN-MCI-AD":
        #     if config.ordered_softmax == False:
        #         x = Conv3D(filters=3, kernel_size=(1, 1, 1), padding='valid')(x)
        #     else:
        #         x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
        # x = Dropout(config.dropout)(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        if config.non_image_data == False:
            # LAST: 1-by-1 convolution + kernel size of 3
            if config.task == "AD" or config.task == "MCI":
                x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
            elif config.task == "CN-MCI-AD":
                if config.ordered_softmax == False:
                    x = Conv3D(filters=3, kernel_size=(1, 1, 1), padding='valid')(x)
                else:
                    x = Conv3D(filters=2, kernel_size=(1, 1, 1), padding='valid')(x)
            x = Dropout(config.dropout)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = GlobalAveragePooling3D()(x)

            # OUTPUT
            if config.ordered_softmax == False:
                predictions = Activation('softmax')(x)
            else:
                predictions = Activation('sigmoid')(x)
            model = Model(inputs=input_image, outputs=predictions)
        elif config.non_image_data == True:
            x = GlobalAveragePooling3D()(x)
            x = keras.layers.concatenate([x, non_image_input])

            # x = Dense(32, activation='relu')(x) # new
            # x = Dense(32, activation='relu')(x)
            # x = Dense(16, activation='relu')(x) # new
            # x = Dense(16, activation='relu')(x)
            # x = Dense(8, activation='relu')(x) # new
            # x = Dense(8, activation='relu')(x)  # new
            # x = Dense(3, activation='relu')(x)

            # x = Dense(16, activation='relu')(x)
            x = Dense(16, activation='relu')(x)
            x = Dense(3, activation='relu')(x)

            # OUTPUT
            if config.ordered_softmax == False:
                mixed_output = Activation('softmax')(x)
            else:
                mixed_output = Activation('sigmoid')(x)
            model = Model(inputs=[input_image, non_image_input], outputs=mixed_output)



        trained_model = load_model(pretrained_model)
        # model.trainable = False
        number_of_layers = len(trained_model.layers)
        for j in range(number_of_layers):
            if model.layers[j].name == "global_average_pooling3d_1":
                # if config.freeze_weights == True:
                #     for k in range(j, number_of_layers):
                #         model.layers[k].trainable = False
                break
            extracted_weights = trained_model.layers[j].get_weights()
            model.layers[j].set_weights(extracted_weights)
            if config.freeze_weights == True:
                model.layers[j].trainable = False



        if config.task == "AD" or config.task == "MCI":
            model.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                        metrics=['accuracy'])
        elif config.task == "CN-MCI-AD":
            if config.lr_optimiser == "Adam":
                if config.loss_function == OCC.customLossFunction:
                    model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std),
                                  optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                                  metrics=['categorical_accuracy'])
                else:
                    model.compile(loss=config.loss_function,
                                  optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                                  metrics=['categorical_accuracy'])
            elif config.lr_optimiser == "Adadelta":
                if config.loss_function == OCC.customLossFunction:
                    model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std), optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                                  metrics=['categorical_accuracy'])
                else:
                    model.compile(loss=config.loss_function, optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                                  metrics=['categorical_accuracy'])

            # keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False), keras.optimizers.Adadelta(lr=1.0, rho=0.95)
            # categorical_crossentropy, categorical_hinge

    else: # if config.only_non_imaging == True
        x = Dense(32, activation='relu')(non_image_input)  # new
        # x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        # x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)  # new
        # x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        # x = BatchNormalization()(x)
        x = Dense(8, activation='relu')(x)  # new
        # x = BatchNormalization()(x)
        x = Dense(8, activation='relu')(x)  # new
        # x = BatchNormalization()(x)
        x = Dense(3, activation='relu')(x)

        # OUTPUT
        if config.ordered_softmax == False:
            output = Activation('softmax')(x)
        else:
            output = Activation('sigmoid')(x)
        model = Model(inputs=non_image_input, outputs=output)

        if config.lr_optimiser == "Adam":
            if config.loss_function == OCC.customLossFunction:
                model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std),
                              optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                              metrics=['categorical_accuracy'])
            else:
                model.compile(loss=config.loss_function,
                              optimizer=keras.optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=config.epsilon, decay=config.decay, amsgrad=False),
                              metrics=['categorical_accuracy'])
        elif config.lr_optimiser == "Adadelta":
            if config.loss_function == OCC.customLossFunction:
                model.compile(loss=config.loss_function(non_image_input, non_image_mean, non_image_std), optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                              metrics=['categorical_accuracy'])
            else:
                model.compile(loss=config.loss_function, optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95),
                              metrics=['categorical_accuracy'])

    return model

