# import configuration file
import config

# set random seed
#from numpy.random import seed
#from tensorflow import set_random_seed
#seed(config.fixed_seed)
#set_random_seed(config.fixed_seed)

# select display for plotting (no display when running on server)
#import os
import matplotlib as mpl
#if os.environ.get('DISPLAY','') == '':
#    print('no display found. Using non-interactive Agg backend')
# Use non-interactive Agg backend
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import auc
from scipy import interp


def plot_acc_loss(history, results_dir, fold, train_callback, val_callback):
    """
    Plots the epoch-wise accuracy and loss based on the history of a DL model.

        INPUT:
            history - training history created with the callbacks
            results_dir - directory where to save the plot
            fold - specify the fold of the cross-validation

        OUTPUT:
            save a plot of the training and validation accuracy, loss and learning rate
    """

    # plot history for accuracy
    if config.task == "AD" or config.task == "MCI":
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
    #elif config.task == "CN-MCI-AD":
    #    plt.plot(history.history['categorical_accuracy'])
    #    plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model accuracy - fold ' + str(fold))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(results_dir + '/acc.png')
        plt.close()

        # plot history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss - fold ' + str(fold))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(results_dir + '/loss.png')
        plt.close()

    elif config.task == "CN-MCI-AD":
        # plot history for loss
        # plt.plot(history.history['loss'])
        plt.plot(train_callback.loss_list)
        # plt.plot(history.history['val_loss'])
        plt.plot(val_callback.loss_list)
        plt.title('Training and validation loss per epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(results_dir + '/loss.png')
        plt.close()

    # plot learning rate
    if config.lr_scheduler:
        plt.plot(history.history['lr'])
        plt.title('Learning rate - fold ' + str(fold))
        plt.ylabel('LR')
        plt.xlabel('epoch')
        plt.legend(['learning rate'], loc='upper left')
        plt.savefig(results_dir + '/lr.png')
        plt.close()



def plot_ROC(total_tpr, total_fpr, total_auc):
    """
    Plots the ROC-AUC curve based on the amount of true and false positives.

        INPUT:
            total_tpr - the true positive rate for every fold
            total_fpr - the false positive rate for every fold
            total_auc - the final AUC for every fold

        OUTPUT:
            plot of the ROC-AUC for all folds of the cross-validation
            including the average AUC and std per fold
            saved in the output dir
    """

    tprs = []

    # plot the AUC for every fold
    for i in range(config.k_cross_validation):
        tprs.append(interp(config.fpr_interval, total_fpr[i], total_tpr[i]))
        tprs[-1][0] = 0.0
        plt.plot(total_fpr[i], total_tpr[i], lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, total_auc[i]))

    # plot the mean AUC
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(config.fpr_interval, mean_tpr)
    std_auc = np.std(total_auc)
    plt.plot(config.fpr_interval, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    # plot std range
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(config.fpr_interval, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    # settings
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(config.output_dir + config.task + '-ROC.pdf')
    plt.close()


def plot_epoch_performance(callback_object):
    """
    Plots the performance stored in the callback object class "EpochPerformance"
    In one plot the sens, spec, auc and acc scores per epoch are visible.

        INPUT:
            callback_object - the epoch information stored in a callback object

        OUTPUT:
            plot of the epoch performance, including sens, spec, AUC and acc
            saves in the results dir of a fold
    """

    # extract from object
    results_dir = callback_object.results_dir
    if config.task == "AD" or config.task == "MCI":
        acc = callback_object.acc_list
        sens = callback_object.sens_list
        spec = callback_object.spec_list
        auc = callback_object.auc_list
    elif config.task == "CN-MCI-AD":
        mauc = callback_object.mauc_list
        bca = callback_object.bca_list
        acc = callback_object.acc_list
        auc_converters = callback_object.auc_list_converters
        bca_converters = callback_object.bca_list_converters
        acc_converters = callback_object.acc_list_converters
        mauc_non_converters = callback_object.mauc_list_non_converters
        bca_non_converters = callback_object.bca_list_non_converters
        acc_non_converters = callback_object.acc_list_non_converters
    if callback_object.name is "trn":
        name = "Training"
    else:
        name = "Validation"

    # plots per epoch
    if config.task == "AD" or config.task == "MCI":
        plt.plot(acc)
        plt.plot(sens)
        plt.plot(spec)
        plt.plot(auc)
    elif config.task == "CN-MCI-AD":
        plt.plot(mauc)
        plt.plot(bca)
        plt.plot(acc_converters)
        plt.plot(bca_converters)
        plt.plot(mauc_non_converters)
        plt.plot(bca_non_converters)

    plt.title(name + ' performance per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    if config.task == "AD" or config.task == "MCI":
        plt.legend(['acc', 'sens', 'spec', 'auc'], loc='lower right')
    elif config.task == "CN-MCI-AD":
        plt.legend(['MAUC', 'BCA', 'ACC-C', 'BCA-C', 'MAUC-NC', 'BCA-NC'], loc='lower right')
    plt.savefig(results_dir + '/' + name + '_performance_per_epoch.png')
    plt.close()
