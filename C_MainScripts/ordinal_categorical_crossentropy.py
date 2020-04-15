from keras import backend as K
from keras import losses
import config
import numpy as np



def customLossFunction(non_image, non_image_mean, non_image_std):

    def loss(y_true, y_pred):

        batch_loss = 0
        if y_true.shape[0] == config.batch_size:
            batch_size = y_true.shape[0]
        else:
            batch_size = 1

        for i in range(batch_size):
            sample_y_true = y_true[i,:]
            sample_y_pred = y_pred[i,:]
            sample_curr_diag = K.cast(non_image[i,0], dtype='float32')
            sample_curr_diag = K.cast(K.round(sample_curr_diag), dtype='float32')

            # Normalise y_true the same way as current diagnosis in the Generator
            sample_y_true_index = K.cast(K.argmax(sample_y_true, axis=0) + 1, dtype='float32')
            if config.differentiate_mci_ad_penalty == True:
                this_converter_penalty = K.switch(K.equal(sample_y_true_index, K.cast(3, dtype='float32')), K.cast(config.converter_penalty_ad, dtype='float32'), K.cast(config.converter_penalty_mci, dtype='float32'))
            sample_y_true_normalised = K.cast(np.subtract(sample_y_true_index, non_image_mean[0]), dtype='float32')
            sample_y_true_normalised = K.cast(np.divide(sample_y_true_normalised, non_image_std[0]), dtype='float32')
            sample_y_true_normalised = K.cast(K.round(sample_y_true_normalised), dtype='float32')

            if config.ordinal_penalty == True:
                ordinal_penalty = K.cast(K.abs(K.argmax(sample_y_true, axis=0) - K.argmax(sample_y_pred, axis=0)) / (K.int_shape(sample_y_pred)[0] - 1), dtype='float32')
            else:
                ordinal_penalty = K.cast(0, dtype='float32')

            if config.extra_penalty_misclassified_converters == True:
                if config.differentiate_mci_ad_penalty == False:
                    converter_penalty = K.switch(K.less(sample_curr_diag, sample_y_true_normalised), K.cast(config.converter_penalty, dtype='float32'), K.cast(1, dtype='float32'))
                else:
                    converter_penalty = K.switch(K.less(sample_curr_diag, sample_y_true_normalised), K.cast(this_converter_penalty, dtype='float32'), K.cast(1, dtype='float32'))
            else:
                converter_penalty = K.cast(1, dtype='float32')

            batch_loss += losses.categorical_crossentropy(sample_y_true, sample_y_pred) * (1 + ordinal_penalty) * converter_penalty

        return batch_loss / batch_size

    return loss


