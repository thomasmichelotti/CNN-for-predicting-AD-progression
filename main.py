# import configuration file
import config

# import main scripts
from C_MainScripts.augmentation import augmentation, augmentation_mix_all, augmentation_mix_only_converters
from C_MainScripts.callbacks import callbacks_list
import C_MainScripts.compute_CI as compute_CI
from C_MainScripts.create_sets import create_labels, split_train_test, split_train_val, format_labels, count_sets, create_sets_for_cv, keep_one_mri_per_individual, non_imaging_sets
from C_MainScripts.generator import DataGenerator, DataGeneratorMultipleInputs, DataGeneratorOnlyNonImagingFeatures, DataGenerator_random_scans, DataGeneratorMultipleInputs_random_scans, DataGeneratorOnlyNonImagingFeatures_random_scans
from C_MainScripts.model_selection import select_model, load_best_model
from C_MainScripts.plotting import plot_acc_loss, plot_ROC, plot_epoch_performance
from C_MainScripts.savings import save_results, save_DL_model
from C_MainScripts.standardize import standardization_matrix, non_image_standardization_matrix
from C_MainScripts.MAUC import MAUC
from C_MainScripts.evalOneSubmissionExtended import calcBCA
import C_MainScripts.ordinal_categorical_crossentropy as OCC
from sklearn.metrics.cluster import adjusted_rand_score


# other imports
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from shutil import copyfile
import time
import pandas as pd
import sys
import shutil
from scipy.special import softmax as sftmx

# set random seed
#from numpy.random import seed
#from tensorflow import set_random_seed
#seed(config.fixed_seed)
#set_random_seed(config.fixed_seed)

import numpy as np
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(config.fixed_seed)
rn.seed(config.fixed_seed)
tf.set_random_seed(config.fixed_seed)

# if config.reproducible == True:
#     from keras import backend as K
#     session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#     sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#     K.set_session(sess)



def main(argv):
    """
    This script will implement the training and testing of a deep learning model for the TADPOLE prediction of AD
    clinical diagnosis progression, based on structural MRI data. All settings and model parameters can be defined
    in the config.py configuration file.

    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    # start timer
    start = time.time()
    start_localtime = time.localtime()

    # if temp job dir is provided as input use this as data dir (server)
    if len(argv) > 1:
        config.data_dir = sys.argv[1] + "/"
        config.aug_dir = sys.argv[1] + "/augmentation/"

    # use the actual opened data 'roi' output folder name
    if len(argv) > 3:
        config.roi = sys.argv[3]

    if len(argv) > 4:
        lb_temp = sys.argv[4]
        ub_temp = sys.argv[5]
        config.comments = config.comments.replace(f"interval_{config.lb}_{config.ub}", f"interval_{lb_temp}_{ub_temp}")
        config.output_dir = config.output_dir.replace(f"interval_{config.lb}_{config.ub}", f"interval_{lb_temp}_{ub_temp}")
        config.aug_dir = config.aug_dir.replace(f"interval_{config.lb}_{config.ub}", f"interval_{lb_temp}_{ub_temp}")
        config.final_output_dir = config.final_output_dir.replace(f"interval_{config.lb}_{config.ub}", f"interval_{lb_temp}_{ub_temp}")
        config.str_out_final = config.str_out_final.replace(f"interval_{config.lb}_{config.ub}", f"interval_{lb_temp}_{ub_temp}")
        config.lb = lb_temp
        config.ub = ub_temp
        config.interval = f"interval_{config.lb}_{config.ub}"

    # save configuration file
    create_data_directory(config.output_dir)
    copyfile(config.config_file, f"{config.output_dir}configuration_{config.model}.py")

    # Different path for AD/MCI classification and for CN-MCI-AD prediction
    if config.task == "AD" or config.task == "MCI":
        # create labels
        partition_labels, labels = create_labels()
        np.save(config.output_dir + "partition_labels.npy", partition_labels)
        np.save(config.output_dir + "labels.npy", labels)

        # train test split
        partition_train_test, test_index = split_train_test(partition_labels, labels)
        np.save(config.output_dir + "train_test.npy", partition_train_test)

        # train val split
        partition_train_validation = split_train_val(partition_train_test, labels)
        np.save(config.output_dir + "train_val.npy", partition_train_validation)

        # initialization of results dictionary
        results = {
            "train": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []},
            "validation": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [],
                           "specificity": []},
            "test": {"loss": [], "acc": [], "fpr": [], "tpr": [], "auc": [], "sensitivity": [], "specificity": []}}

        # START CROSS VALIDATION
        for i in range(config.k_cross_validation):

            # select model
            model = select_model(i)

            print("\n----------- CROSS VALIDATION " + str(i) + " ----------------\n")

            # augmentation of training data
            if config.augmentation:
                partition_train_validation["train"][i], labels = augmentation(partition_train_validation["train"][i],
                                                                              labels)
                count_sets(partition_train_validation, labels)

            # create results directory for fold
            results_dir = config.output_dir + "k" + str(i)
            create_data_directory(results_dir)
            file = open(results_dir + "/results.txt", 'w')

            # get mean + std of train data to standardize all data in generator
            if config.all_data:
                mean = np.load(config.mean_file)
                std = np.load(config.std_file)
            else:
                mean, std = standardization_matrix(partition_train_validation["train"][i])

            # save mean + std
            np.save(results_dir + "/mean.npy", mean)
            np.save(results_dir + "/std.npy", std)

            # create data generators
            train_generator = DataGenerator(partition_train_validation["train"][i], labels, mean, std,
                                            batch_size=config.batch_size, dim=config.input_shape, n_channels=1,
                                            n_classes=2, shuffle=True)
            validation_generator = DataGenerator(partition_train_validation["validation"][i], labels, mean, std,
                                                 batch_size=config.batch_size, dim=config.input_shape, n_channels=1,
                                                 n_classes=2, shuffle=True)
            test_generator = DataGenerator(partition_train_test["test"][i], labels, mean, std,
                                           batch_size=1, dim=config.input_shape, n_channels=1, n_classes=2,
                                           shuffle=False)
            # generators used for evaluation (CM for confusion matrix), batch size set to 1
            CM_train_generator = DataGenerator(partition_train_validation["train"][i], labels, mean, std,
                                               batch_size=1, dim=config.input_shape, n_channels=1, n_classes=2,
                                               shuffle=False)
            CM_validation_generator = DataGenerator(partition_train_validation["validation"][i], labels, mean, std,
                                                    batch_size=1, dim=config.input_shape, n_channels=1, n_classes=2,
                                                    shuffle=False)

            # set callbacks
            callback_list = callbacks_list(CM_train_generator, CM_validation_generator, labels, results_dir)

            if not config.test_only:

                # TRAINING
                history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                              class_weight=None, callbacks=callback_list, epochs=config.epochs,
                                              verbose=1, use_multiprocessing=False, workers=0)

                # plot acc + loss
                plot_acc_loss(history, results_dir, i)

                # plot performance per epoch
                if config.epoch_performance:
                    plot_epoch_performance(callback_list[0])
                    plot_epoch_performance(callback_list[1])

                # load model of epoch with best performance
                model = load_best_model(results_dir)

                # TRAIN EVALUATION

                # roc auc
                Y_pred = model.predict_generator(CM_train_generator, verbose=0)
                y_pred = np.argmax(Y_pred, axis=1)
                y_true = []
                for id in CM_train_generator.list_IDs:
                    y_true.append(labels[id])
                fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:, 1])
                roc_auc = auc(fpr, tpr)

                # save classification per subject (for statistical test)
                np.save(results_dir + "/train_IDs.npy", CM_train_generator.list_IDs)
                np.save(results_dir + "/train_y_true.npy", y_true)
                np.save(results_dir + "/train_y_pred.npy", y_pred)

                # sen / spe
                report = classification_report(y_true, y_pred, target_names=[config.class0, config.class1],
                                               output_dict=True)

                # loss, acc
                score = model.evaluate_generator(generator=train_generator, verbose=1)

                results["train"]["loss"].append(score[0])
                results["train"]["acc"].append(score[1])
                results["train"]["fpr"].append(fpr)
                results["train"]["tpr"].append(tpr)
                results["train"]["auc"].append(roc_auc)
                results["train"]["sensitivity"].append(report[config.class1]["recall"])
                results["train"]["specificity"].append(report[config.class0]["recall"])

                # report train results
                train_results = f"\nTrain\n    loss: {score[0]:.4f}\n    acc: {score[1]:.4f}\n    AUC: {roc_auc:.4f}\n    " \
                                f"sens: {report[config.class1]['recall']:.4f}\n    spec: {report[config.class0]['recall']:.4f}\n\n"
                file.write(train_results), print(train_results)

                # VALIDATION EVALUATION

                # roc auc
                Y_pred = model.predict_generator(CM_validation_generator, verbose=0)
                y_pred = np.argmax(Y_pred, axis=1)
                y_true = []
                for id in CM_validation_generator.list_IDs:
                    y_true.append(labels[id])
                fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:, 1])
                roc_auc = auc(fpr, tpr)

                # save classification per subject (for statistical test)
                np.save(results_dir + "/val_IDs.npy", CM_validation_generator.list_IDs)
                np.save(results_dir + "/val_y_true.npy", y_true)
                np.save(results_dir + "/val_y_pred.npy", y_pred)

                # sen / spe
                report = classification_report(y_true, y_pred, target_names=[config.class0, config.class1],
                                               output_dict=True)

                # loss, acc
                score = model.evaluate_generator(generator=validation_generator, verbose=1)

                results["validation"]["loss"].append(score[0])
                results["validation"]["acc"].append(score[1])
                results["validation"]["fpr"].append(fpr)
                results["validation"]["tpr"].append(tpr)
                results["validation"]["auc"].append(roc_auc)
                results["validation"]["sensitivity"].append(report[config.class1]["recall"])
                results["validation"]["specificity"].append(report[config.class0]["recall"])

                # report val results
                val_results = f"\nValidation\n    loss: {score[0]:.4f}\n    acc: {score[1]:.4f}\n    AUC: {roc_auc:.4f}\n    " \
                              f"sens: {report[config.class1]['recall']:.4f}\n    spec: {report[config.class0]['recall']:.4f}\n\n"
                file.write(val_results), print(val_results)

            # TEST EVALUATION

            # roc auc
            Y_pred = model.predict_generator(test_generator, verbose=0)
            y_pred = np.argmax(Y_pred, axis=1)
            y_true = []
            for id in test_generator.list_IDs:
                y_true.append(labels[id])
            fpr, tpr, thresholds = roc_curve(y_true, Y_pred[:, 1])
            roc_auc = auc(fpr, tpr)

            # save classification per subject (for statistical test)
            np.save(results_dir + "/test_IDs.npy", test_generator.list_IDs)
            np.save(results_dir + "/test_y_true.npy", y_true)
            np.save(results_dir + "/test_y_pred.npy", y_pred)

            # sen / spe
            report = classification_report(y_true, y_pred, target_names=[config.class0, config.class1],
                                           output_dict=True)

            # loss, acc
            score = model.evaluate_generator(generator=test_generator, verbose=1)

            results["test"]["loss"].append(score[0])
            results["test"]["acc"].append(score[1])
            results["test"]["fpr"].append(fpr)
            results["test"]["tpr"].append(tpr)
            results["test"]["auc"].append(roc_auc)
            results["test"]["sensitivity"].append(report[config.class1]["recall"])
            results["test"]["specificity"].append(report[config.class0]["recall"])

            # report test results
            test_results = f"\nTest\n    loss: {score[0]:.4f}\n    acc: {score[1]:.4f}\n    AUC: {roc_auc:.4f}\n    " \
                           f"sens: {report[config.class1]['recall']:.4f}\n    spec: {report[config.class0]['recall']:.4f}\n\n"
            file.write(test_results), print(test_results)
            file.close()

            # delete augmented images
            if config.augmentation:
                os.system('rm -rf %s/*' % config.aug_dir)
                #shutil.rmtree(config.aug_dir)

        print("\n---------------------- RESULTS ----------------------\n\n")

        # plot test ROC of all folds + average + std
        plot_ROC(results["test"]["tpr"], results["test"]["fpr"], results["test"]["auc"])

        # end timer
        end = time.time()
        end_localtime = time.localtime()

        # save results + model
        np.save(config.output_dir + "results.npy", results)
        save_DL_model(model)
        save_results(results, start, start_localtime, end, end_localtime)

        # Confidence interval computation using corrected resampled t-test (Nadeau and Bengio, Machine Learning, 2003)
        N_1 = float(len(CM_train_generator.list_IDs) + len(CM_validation_generator.list_IDs))
        N_2 = float(len(test_generator.list_IDs))
        print(f"Train+validation size: {N_1}, Test size: {N_2}")
        alpha = 0.95

        # Add Test results summary
        test_results = {}
        test_results['acc_mean'] = np.mean(results['test']['acc'])
        test_results['acc_95ci'] = compute_CI.compute_confidence(results['test']['acc'], N_1, N_2, alpha)
        test_results['auc_mean'] = np.mean(results['test']['auc'])
        test_results['auc_95ci'] = compute_CI.compute_confidence(results['test']['auc'], N_1, N_2, alpha)

        print('mean Acc:', '%.3f' % test_results['acc_mean'], ' (', '%.3f' % test_results['acc_95ci'][0], ' - ',
              '%.3f' % test_results['acc_95ci'][1], ')')
        print('mean AUC:', '%.3f' % test_results['auc_mean'], ' (', '%.3f' % test_results['auc_95ci'][0], ' - ',
              '%.3f' % test_results['auc_95ci'][1], ')')

        np.save(config.output_dir + "test_results.npy", test_results)

        print('\nend')



    elif config.task == "CN-MCI-AD":

        if config.TADPOLE_setup == True:
            if config.interval_CNN == False:
                if config.leaderboard:
                    train_labels = pd.read_csv(config.preprocessed_data_dir + "Leaderboard/D2/Train/labels.csv")
                    validation_labels = pd.read_csv(config.preprocessed_data_dir + "Leaderboard/D2/Validation/labels.csv")
                    test_labels = pd.read_csv(config.preprocessed_data_dir + "Leaderboard/D2/Test/labels.csv")
                    if config.non_image_data:
                        train_features = pd.read_csv(config.preprocessed_data_dir + "Leaderboard/D2/Train/" + config.non_imaging_filename)
                        validation_features = pd.read_csv(config.preprocessed_data_dir + "Leaderboard/D2/Validation/" + config.non_imaging_filename)
                        test_features = pd.read_csv(config.preprocessed_data_dir + "Leaderboard/D2/Test/" + config.non_imaging_filename)
                else:
                    train_labels = pd.read_csv(config.preprocessed_data_dir + "TADPOLE/D2/Train/labels.csv")
                    validation_labels = pd.read_csv(config.preprocessed_data_dir + "TADPOLE/D2/Validation/labels.csv")
                    test_labels = pd.read_csv(config.preprocessed_data_dir + "TADPOLE/D2/Test/labels.csv")
                    if config.non_image_data:
                        train_features = pd.read_csv(config.preprocessed_data_dir + "TADPOLE/D2/Train/" + config.non_imaging_filename)
                        validation_features = pd.read_csv(config.preprocessed_data_dir + "TADPOLE/D2/Validation/" + config.non_imaging_filename)
                        test_features = pd.read_csv(config.preprocessed_data_dir + "TADPOLE/D2/Test/" + config.non_imaging_filename)
            else: #config.interval_CNN == True
                if config.leaderboard:
                    train_labels = pd.read_csv(config.preprocessed_data_dir + f"Leaderboard/Intervals/Train/{config.interval}/labels.csv")
                    validation_labels = pd.read_csv(config.preprocessed_data_dir + f"Leaderboard/Intervals/Validation/{config.interval}/labels.csv")
                    test_labels = pd.read_csv(config.preprocessed_data_dir + f"Leaderboard/Intervals/Test/{config.interval}/labels.csv")
                    if config.non_image_data:
                        train_features = pd.read_csv(config.preprocessed_data_dir + f"Leaderboard/Intervals/Train/{config.interval}/" + config.non_imaging_filename)
                        validation_features = pd.read_csv(config.preprocessed_data_dir + f"Leaderboard/Intervals/Validation/{config.interval}/" + config.non_imaging_filename)
                        test_features = pd.read_csv(config.preprocessed_data_dir + f"Leaderboard/Intervals/Test/{config.interval}/" + config.non_imaging_filename)
                else: # DOESN'T WORK YET!!!
                    train_labels = pd.read_csv(config.preprocessed_data_dir + f"TADPOLE/Intervals/Train/{config.interval}/labels.csv")
                    validation_labels = pd.read_csv(config.preprocessed_data_dir + f"TADPOLE/Intervals/Validation/{config.interval}/labels.csv")
                    test_labels = pd.read_csv(config.preprocessed_data_dir + f"TADPOLE/Intervals/Test/{config.interval}/labels.csv")
                    if config.non_image_data:
                        train_features = pd.read_csv(config.preprocessed_data_dir + f"TADPOLE/Intervals/Train/{config.interval}/" + config.non_imaging_filename)
                        validation_features = pd.read_csv(config.preprocessed_data_dir + f"TADPOLE/Intervals/Validation/{config.interval}/" + config.non_imaging_filename)
                        test_features = pd.read_csv(config.preprocessed_data_dir + f"TADPOLE/Intervals/Test/{config.interval}/" + config.non_imaging_filename)


            # create labels and split train-val-test
            partition_labels, labels, partition_train_val_test = format_labels(train_labels, validation_labels, test_labels)
            np.save(config.output_dir + "partition_labels.npy", partition_labels)
            np.save(config.output_dir + "labels.npy", labels)
            np.save(config.output_dir + "partition_train_val_test.npy", partition_train_val_test)


            # initialization of results dictionary
            results = {"train": {"loss": [], "mauc": [], "bca": []},
                       "validation": {"loss": [], "mauc": [], "bca": []},
                       "test": {"loss": [], "mauc": [], "bca": []}} # no loss in test set

            i = 0 # No CV used, so everything will be in the "first fold"
            # select model
            model = select_model(i)


            # create results directory
            results_dir = config.output_dir + "results"
            create_data_directory(results_dir)
            file = open(results_dir + "/results.txt", 'w')


            # get mean + std of train data to standardize all data in generator
            if config.only_non_imaging == False:
                if config.all_data:
                    mean = np.load(config.mean_file)
                    std = np.load(config.std_file)
                else:
                    mean, std = standardization_matrix(partition_train_val_test["train"])


            # get mean + std of non-image train data to standardize non-image data in generator
            if config.non_image_data == True:
                non_image_mean, non_image_std = non_image_standardization_matrix(train_features)


            # save mean + std
            if config.only_non_imaging == False:
                np.save(results_dir + "/mean.npy", mean)
                np.save(results_dir + "/std.npy", std)
            if config.non_image_data == True:
                np.save(results_dir + "/non_image_mean.npy", non_image_mean)
                np.save(results_dir + "/non_image_std.npy", non_image_std)


            # create data generators
            if config.non_image_data == False:
                train_generator = DataGenerator(partition_train_val_test["train"], labels, mean, std,
                                                batch_size=config.batch_size, dim=config.input_shape, n_channels=1, n_classes=3,
                                                shuffle=True)
                validation_generator = DataGenerator(partition_train_val_test["validation"], labels, mean, std,
                                                     batch_size=config.batch_size, dim=config.input_shape, n_channels=1,
                                                     n_classes=3, shuffle=True)
                CM_train_generator = DataGenerator(partition_train_val_test["train"], labels, mean, std,
                                                   batch_size=1, dim=config.input_shape, n_channels=1, n_classes=3,
                                                   shuffle=False)
                CM_validation_generator = DataGenerator(partition_train_val_test["validation"], labels, mean, std,
                                                        batch_size=1, dim=config.input_shape, n_channels=1, n_classes=3,
                                                        shuffle=False)
                CM_test_generator = DataGenerator(partition_train_val_test["test"], labels, mean, std,
                                                        batch_size=1, dim=config.input_shape, n_channels=1, n_classes=3,
                                                        shuffle=False)
            elif config.non_image_data == True:
                dim_non_imaging_features = (config.non_image_input_shape, )
                if config.only_non_imaging == False:
                    train_generator = DataGeneratorMultipleInputs(partition_train_val_test["train"], labels, train_features, mean, std, non_image_mean, non_image_std,
                                                    batch_size=config.batch_size, dim=config.input_shape, dim_non_image=dim_non_imaging_features,
                                                    n_channels=1, n_classes=3, shuffle=True)
                    validation_generator = DataGeneratorMultipleInputs(partition_train_val_test["validation"], labels, validation_features, mean, std, non_image_mean, non_image_std,
                                                         batch_size=config.batch_size, dim=config.input_shape, dim_non_image=dim_non_imaging_features,
                                                    n_channels=1, n_classes=3, shuffle=True)
                    CM_train_generator = DataGeneratorMultipleInputs(partition_train_val_test["train"], labels, train_features, mean, std, non_image_mean, non_image_std,
                                                       batch_size=1, dim=config.input_shape, dim_non_image=dim_non_imaging_features,
                                                    n_channels=1, n_classes=3, shuffle=False)
                    CM_validation_generator = DataGeneratorMultipleInputs(partition_train_val_test["validation"], labels, validation_features, mean, std, non_image_mean, non_image_std,
                                                            batch_size=1, dim=config.input_shape, dim_non_image=dim_non_imaging_features,
                                                    n_channels=1, n_classes=3, shuffle=False)
                    CM_test_generator = DataGeneratorMultipleInputs(partition_train_val_test["test"], labels, test_features, mean, std, non_image_mean, non_image_std,
                                                      batch_size=1, dim=config.input_shape, dim_non_image=dim_non_imaging_features,
                                                    n_channels=1, n_classes=3, shuffle=False)
                else: # if config.only_non_imaging == True
                    train_generator = DataGeneratorOnlyNonImagingFeatures(partition_train_val_test["train"], labels, train_features, non_image_mean, non_image_std,
                                                                  batch_size=config.batch_size, dim_non_image=dim_non_imaging_features,
                                                                  n_channels=1, n_classes=3, shuffle=True)
                    validation_generator = DataGeneratorOnlyNonImagingFeatures(partition_train_val_test["validation"], labels, validation_features, non_image_mean, non_image_std,
                                                                       batch_size=config.batch_size, dim_non_image=dim_non_imaging_features,
                                                                       n_channels=1, n_classes=3, shuffle=True)
                    CM_train_generator = DataGeneratorOnlyNonImagingFeatures(partition_train_val_test["train"], labels, train_features, non_image_mean, non_image_std,
                                                                     batch_size=1, dim_non_image=dim_non_imaging_features,
                                                                     n_channels=1, n_classes=3, shuffle=False)
                    CM_validation_generator = DataGeneratorOnlyNonImagingFeatures(partition_train_val_test["validation"], labels, validation_features, non_image_mean, non_image_std,
                                                                          batch_size=1, dim_non_image=dim_non_imaging_features,
                                                                          n_channels=1, n_classes=3, shuffle=False)
                    CM_test_generator = DataGeneratorOnlyNonImagingFeatures(partition_train_val_test["test"], labels, test_features, non_image_mean, non_image_std,
                                                                    batch_size=1, dim_non_image=dim_non_imaging_features,
                                                                    n_channels=1, n_classes=3, shuffle=False)

            # set callbacks
            callback_list = callbacks_list(CM_train_generator, CM_validation_generator, labels, results_dir)

            if not config.test_only:

                # TRAINING
                history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                                class_weight=None, callbacks=callback_list, epochs=config.epochs,
                                                verbose=1, use_multiprocessing=False, workers=0)

                # plot acc + loss
                plot_acc_loss(history, results_dir, i)

                # plot performance per epoch
                if config.epoch_performance:
                    plot_epoch_performance(callback_list[0]) # training performance
                    plot_epoch_performance(callback_list[1]) # validation performance
                # load model of epoch with best performance
                model = load_best_model(results_dir)

                # TRAINING EVALUATION
                print("\n---------------------- TRAINING EVALUATION ----------------------\n\n")

                # mauc, bca
                Y_pred = model.predict_generator(CM_train_generator, verbose=0)
                y_pred = np.argmax(Y_pred, axis=1)
                y_true = []
                for id in CM_train_generator.list_IDs:
                    y_true.append(labels[id])

                # calculate MAUC based on prediction estimates
                nrSubj = Y_pred.shape[0]
                nrClasses = Y_pred.shape[1]
                hardEstimClass = -1 * np.ones(nrSubj, int)
                zipTrueLabelAndProbs = []
                for s in range(nrSubj):
                    pCN = Y_pred[s, 0]
                    pMCI = Y_pred[s, 1]
                    pAD = Y_pred[s, 2]
                    hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                    zipTrueLabelAndProbs += [(y_true[s], [pCN, pMCI, pAD])]
                zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                mauc = MAUC(zipTrueLabelAndProbs, nrClasses)

                # calculate BCA based on prediction estimates
                true_labels = np.asarray(y_true)
                bca = calcBCA(hardEstimClass, true_labels, nrClasses)
                conf = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                file.write("Confusion matrix TRAINING set:\n")
                file.write(np.array2string(conf, separator=', '))
                print(conf)

                # loss, acc
                score = model.evaluate_generator(generator=train_generator, verbose=1)
                #score_loss_in_CM_train_generator = model.evaluate_generator(generator=CM_train_generator, verbose=1)

                results["train"]["loss"].append(score[0])
                results["train"]["mauc"].append(mauc)
                results["train"]["bca"].append(bca)

                # save classification per subject (for statistical test)
                np.save(results_dir + "/train_IDs.npy", CM_train_generator.list_IDs)
                np.save(results_dir + "/train_y_true.npy", y_true)
                np.save(results_dir + "/train_y_pred.npy", y_pred)

                # report train results
                train_results = f"\nTrain\n    loss: {score[0]:.4f}\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    "
                file.write(train_results), print(train_results)

                # VALIDATION EVALUATION
                print("\n---------------------- VALIDATION EVALUATION ----------------------\n\n")

                # mauc, bca
                if config.non_image_data == False:
                    Y_pred = model.predict_generator(CM_validation_generator, verbose=0)
                elif config.non_image_data == True: # CHANGE!
                    Y_pred = model.predict_generator(CM_validation_generator, verbose=0)
                y_pred = np.argmax(Y_pred, axis=1)
                y_true = []
                for id in CM_validation_generator.list_IDs:
                    y_true.append(labels[id])

                # calculate MAUC based on prediction estimates
                nrSubj = Y_pred.shape[0]
                nrClasses = Y_pred.shape[1]
                hardEstimClass = -1 * np.ones(nrSubj, int)
                zipTrueLabelAndProbs = []
                for s in range(nrSubj):
                    pCN = Y_pred[s, 0]
                    pMCI = Y_pred[s, 1]
                    pAD = Y_pred[s, 2]
                    hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                    zipTrueLabelAndProbs += [(y_true[s], [pCN, pMCI, pAD])]
                zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                mauc = MAUC(zipTrueLabelAndProbs, nrClasses)

                # calculate BCA based on prediction estimates
                true_labels = np.asarray(y_true)
                bca = calcBCA(hardEstimClass, true_labels, nrClasses)
                conf = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                file.write("Confusion matrix VALIDATION set:\n")
                file.write(np.array2string(conf, separator=', '))
                print(conf)

                # loss, acc
                score = model.evaluate_generator(generator=validation_generator, verbose=1)

                results["validation"]["loss"].append(score[0])
                results["validation"]["mauc"].append(mauc)
                results["validation"]["bca"].append(bca)

                # save classification per subject (for statistical test)
                np.save(results_dir + "/val_IDs.npy", CM_validation_generator.list_IDs)
                np.save(results_dir + "/val_y_true.npy", y_true)
                np.save(results_dir + "/val_y_pred.npy", y_pred)

                # report val results
                val_results = f"\nValidation\n    loss: {score[0]:.4f}\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    "
                file.write(val_results), print(val_results)
                #file.close()

                # delete augmented images
                if config.augmentation:
                    os.system('rm -rf %s/*' % config.aug_dir)
                    # shutil.rmtree(config.aug_dir)

            # TESTING EVALUATION
            print("\n---------------------- TESTING EVALUATION ----------------------\n\n")

            # make predictions for test set (either LB4 or D4)
            if config.non_image_data == False:
                Y_pred = model.predict_generator(CM_test_generator, verbose=0)
            elif config.non_image_data == True:  # CHANGE!
                Y_pred = model.predict_generator(CM_test_generator, verbose=0)

            if config.leaderboard:
                # mauc and bca
                y_pred = np.argmax(Y_pred, axis=1)
                y_true = []
                for id in CM_test_generator.list_IDs:
                    y_true.append(labels[id])
                # calculate MAUC based on prediction estimates
                nrSubj = Y_pred.shape[0]
                nrClasses = Y_pred.shape[1]
                hardEstimClass = -1 * np.ones(nrSubj, int)
                zipTrueLabelAndProbs = []
                for s in range(nrSubj):
                    pCN = Y_pred[s, 0]
                    pMCI = Y_pred[s, 1]
                    pAD = Y_pred[s, 2]
                    hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                    zipTrueLabelAndProbs += [(y_true[s], [pCN, pMCI, pAD])]
                zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                mauc = MAUC(zipTrueLabelAndProbs, nrClasses)

                # calculate BCA based on prediction estimates
                true_labels = np.asarray(y_true)
                bca = calcBCA(hardEstimClass, true_labels, nrClasses)
                conf = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                file.write("Confusion matrix TEST set:\n")
                file.write(np.array2string(conf, separator=', '))
                print(conf)

                # loss, acc
                #score = model.evaluate_generator(generator=test_generator, verbose=1) # why is this not CM_test_generator? Why need 2?

                #results["test"]["loss"].append(score[0]) # NO LOSS FOR TESTING, ONLY MAUC AND BCA?
                results["test"]["mauc"].append(mauc)
                results["test"]["bca"].append(bca)

                # save classification per subject (for statistical test)
                np.save(results_dir + "/test_IDs.npy", CM_test_generator.list_IDs)
                np.save(results_dir + "/test_y_true.npy", y_true)
                np.save(results_dir + "/test_y_pred.npy", y_pred)

                # report test results
                test_results = f"\nTest\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    "
                file.write(test_results), print(test_results)

            file.close()

            nrSubj = Y_pred.shape[0]
            RID = test_labels["RID"].to_numpy()
            y_ADAS13, y_ADAS13_lower, y_ADAS13_upper, y_Ventricles_ICV, y_Ventricles_ICV_lower, y_Ventricles_ICV_upper \
                = np.zeros((6, nrSubj))
            o = np.column_stack((RID, RID, RID, Y_pred, y_ADAS13, y_ADAS13_lower, y_ADAS13_upper, y_Ventricles_ICV,
                                 y_Ventricles_ICV_lower, y_Ventricles_ICV_upper))

            count = 0
            if config.leaderboard:
                years = [str(a) for a in range(2010, 2018)]
            else:
                years = [str(a) for a in range(2018, 2023)]
            months = [str(a).zfill(2) for a in range(1, 13)]
            ym = [y + '-' + mo for y in years for mo in months]
            if config.leaderboard:
                ym = ym[4:-8]
            nrMonths = len(ym)

            o1 = np.zeros((o.shape[0] * nrMonths, o.shape[1]))
            ym1 = [a for b in range(0, len(RID)) for a in ym]
            for i in range(len(o)):
                o1[count:count + nrMonths] = o[i]
                o1[count:count + nrMonths, 1] = range(1, nrMonths + 1)
                count = count + nrMonths

            # Save output
            output = pd.DataFrame(o1, columns=['RID', 'Forecast Month', 'Forecast Date', 'CN relative probability',
                                               'MCI relative probability', 'AD relative probability', 'ADAS13',
                                               'ADAS13 50% CI lower', 'ADAS13 50% CI upper', 'Ventricles_ICV',
                                               'Ventricles_ICV 50% CI lower', 'Ventricles_ICV 50% CI upper'])
            output['Forecast Month'] = output['Forecast Month'].astype(int)
            output['Forecast Date'] = ym1

            create_data_directory(config.final_output_dir)
            output.to_csv(config.str_out_final, header=True, index=False)


            # plot test ROC of all folds + average + std
            #plot_ROC(results["test"]["tpr"], results["test"]["fpr"], results["test"]["auc"]) # CHANGE FOR MULTICLASS?

            # end timer
            end = time.time()
            end_localtime = time.localtime()

            # save results + model
            np.save(config.output_dir + "results.npy", results)
            save_DL_model(model)
            save_results(results, start, start_localtime, end, end_localtime)


        else: #config.TADPOLE_setup == False
            if config.interval_CNN == False:
                print("You need to specify an interval for this task")
            else:  # if config.interval_CNN == True
                all_labels = pd.read_csv(config.preprocessed_data_dir + f"newPatients/{config.interval}/labels.csv")
                #all_labels = all_labels.iloc[0:800,:] #X
                if config.non_image_data:
                    all_features = pd.read_csv(config.preprocessed_data_dir + f"newPatients/{config.interval}/" + config.non_imaging_filename)
                    #all_features = all_features.iloc[0:800,:] #X

            # create labels
            partition_labels, labels_link, labels_ptid_viscode, partition_train_test, partition_train_validation = create_sets_for_cv(all_labels)

            np.save(config.output_dir + "partition_labels.npy", partition_labels)
            np.save(config.output_dir + "labels_link.npy", labels_link)
            np.save(config.output_dir + "labels_ptid_viscode.npy", labels_ptid_viscode)
            np.save(config.output_dir + "train_test.npy", partition_train_test)
            np.save(config.output_dir + "train_val.npy", partition_train_validation)


            # initialization of results dictionary
            results = {"train": {"loss": [], "mauc": [], "bca": [], "acc": [], "ari": [], "auc_converters": [], "bca_converters": [], "acc_converters": [], "ari_converters": [], "mauc_non_converters": [], "bca_non_converters": [], "acc_non_converters": [], "ari_non_converters": []},
                       "validation": {"loss": [], "mauc": [], "bca": [], "acc": [], "ari": [], "auc_converters": [], "bca_converters": [], "acc_converters": [], "ari_converters": [], "mauc_non_converters": [], "bca_non_converters": [], "acc_non_converters": [], "ari_non_converters": []},
                       "test": {"loss": [], "mauc": [], "bca": [], "acc": [], "ari": [], "auc_converters": [], "bca_converters": [], "acc_converters": [], "ari_converters": [], "mauc_non_converters": [], "bca_non_converters": [], "acc_non_converters": [], "ari_non_converters": []}}

            # START CROSS VALIDATION
            for i in range(config.k_cross_validation):

                print("\n----------- CROSS VALIDATION " + str(i) + " ----------------\n")

                # Keep one mri scan per individual
                partition_train_test_unique, partition_train_validation_unique, train_validation_ptid_viscode, train_test_ptid_viscode = keep_one_mri_per_individual(partition_train_test, partition_train_validation, all_labels, i)

                # create the same train-val-test split for non-imaging data
                if config.non_image_data == True:
                    train_features, validation_features, test_features = non_imaging_sets(partition_train_validation, partition_train_test, all_features, i)

                train_set_extra_penalty = all_labels[all_labels["PTID_VISCODE"].isin(train_validation_ptid_viscode["train"][i])]
                unique_train_rid = train_set_extra_penalty.drop_duplicates("PTID_VISCODE", keep="last")
                unique_train_rid_count = unique_train_rid.shape[0]
                print("Unique train RID: ", unique_train_rid_count)
                unique_converters = unique_train_rid["Converter"].sum()

                # check ratio converters/non-converters and set penalty parameter
                if config.extra_penalty_misclassified_converters == True:
                    unique_non_converters = unique_train_rid_count - unique_converters
                    config.converter_penalty = unique_non_converters / unique_converters
                    print(config.converter_penalty)

                    converter_train_set = unique_train_rid[unique_train_rid["Converter"] == True]
                    print(converter_train_set.shape)
                    mci_converters = converter_train_set[converter_train_set["Diagnosis"] == 2]
                    mci_converter_count = mci_converters.shape[0]
                    print(mci_converter_count)
                    mci_percentage = mci_converter_count / unique_converters
                    ad_converters = converter_train_set[converter_train_set["Diagnosis"] == 3]
                    ad_converter_count = ad_converters.shape[0]
                    print(ad_converter_count)
                    ad_percentage = ad_converter_count / unique_converters

                    A = [[ad_percentage, mci_percentage], [ad_converter_count/mci_converter_count, -1]]
                    A = np.asmatrix(A)
                    b = [[config.converter_penalty], [0]]
                    b = np.asmatrix(b)
                    ans = np.linalg.solve(A,b)

                    config.converter_penalty_mci = ans[1,0]
                    print(config.converter_penalty_mci)
                    config.converter_penalty_ad = ans[0,0]
                    print(config.converter_penalty_ad)

                if config.dynamic_batch_size == True:
                    config.batch_size = int(np.ceil(unique_train_rid_count/unique_converters))
                    print("New batch size: ", config.batch_size)

                # augmentation of training data
                if config.augmentation:
                    start_augmentation = time.time()
                    if config.non_image_data == False:
                        train_features = 0
                    train_validation_ptid_viscode["train"][i], labels_ptid_viscode, train_features = augmentation_mix_only_converters(train_validation_ptid_viscode["train"][i], partition_train_validation_unique["train"][i], labels_ptid_viscode, train_features, all_labels)
                    end_augmentation = time.time()
                    print("Elapsed time for data augmentation: ", end_augmentation - start_augmentation)

                # create results directory for fold
                results_dir = config.output_dir + "k" + str(i)
                create_data_directory(results_dir)
                file = open(results_dir + "/results.txt", 'w')

                # # save weight initialisation
                np.save(results_dir + "/partition_train_test_unique.npy", partition_train_test_unique)
                np.save(results_dir + "/partition_train_validation_unique.npy", partition_train_validation_unique)
                np.save(results_dir + "/train_validation_ptid_viscode.npy", train_validation_ptid_viscode)
                np.save(results_dir + "/train_test_ptid_viscode.npy", train_test_ptid_viscode)
                np.save(results_dir + "/labels_ptid_viscode.npy", labels_ptid_viscode)
                if config.non_image_data == True:
                    np.save(results_dir + "/train_features.npy", train_features)
                    np.save(results_dir + "/validation_features.npy", validation_features)
                    np.save(results_dir + "/test_features.npy", test_features)

                # get mean + std of train data to standardize all data in generator
                if config.only_non_imaging == False:
                    if config.all_data:
                        mean = np.load(config.mean_file)
                        std = np.load(config.std_file)
                    else:
                        mean, std = standardization_matrix(train_validation_ptid_viscode["train"][i])

                # get mean + std of non-image train data to standardize non-image data in generator
                if config.non_image_data == True:
                    train_features_no_duplicates = train_features.drop_duplicates("PTID_VISCODE", keep="last")
                    non_image_mean, non_image_std = non_image_standardization_matrix(train_features_no_duplicates)
                else:
                    non_image_mean, non_image_std = [0,0]

                # save mean + std
                if config.only_non_imaging == False:
                    np.save(results_dir + "/mean.npy", mean)
                    np.save(results_dir + "/std.npy", std)
                if config.non_image_data == True:
                    np.save(results_dir + "/non_image_mean.npy", non_image_mean)
                    np.save(results_dir + "/non_image_std.npy", non_image_std)


                count_retrain_patience = 0
                config.retrain = True
                config.weird_predictions = False
                for retrain in range(config.retrain_patience):
                    if config.retrain == True:
                        if count_retrain_patience > 0:
                            destination = results_dir + f"/old_improvements_attempt_{count_retrain_patience}"
                            create_data_directory(destination)
                            for files in os.listdir(results_dir):
                                if "z-e" in files or "weight_initialisation" in files:
                                    shutil.move(results_dir + "/" + files, destination)

                        # select model
                        model = select_model(i, non_image_mean, non_image_std)

                        # save weight initialisation
                        model.save_weights(results_dir + '/weight_initialisation.hdf5')
                        if i == 0 and count_retrain_patience == 0:
                            save_DL_model(model)


                        # create data generators
                        if config.non_image_data == False:
                            train_generator = DataGenerator_random_scans(train_validation_ptid_viscode["train"][i], partition_train_validation_unique["train"][i], labels_ptid_viscode, all_labels, mean, std, batch_size=config.batch_size, dim=config.input_shape, n_channels=1,
                                                            n_classes=3, shuffle=True)
                            validation_generator = DataGenerator_random_scans(train_validation_ptid_viscode["validation"][i], partition_train_validation_unique["validation"][i], labels_ptid_viscode, all_labels, mean, std, batch_size=config.batch_size, dim=config.input_shape,
                                                                 n_channels=1, n_classes=3, shuffle=True)
                            CM_train_generator = DataGenerator_random_scans(train_validation_ptid_viscode["train"][i], partition_train_validation_unique["train"][i], labels_ptid_viscode, all_labels, mean, std, batch_size=1, dim=config.input_shape, n_channels=1, n_classes=3,
                                                               shuffle=False)
                            CM_validation_generator = DataGenerator_random_scans(train_validation_ptid_viscode["validation"][i], partition_train_validation_unique["validation"][i], labels_ptid_viscode, all_labels, mean, std, batch_size=1, dim=config.input_shape, n_channels=1,
                                                                    n_classes=3, shuffle=False)
                            CM_test_generator = DataGenerator_random_scans(train_test_ptid_viscode["test"][i], partition_train_test_unique["test"][i], labels_ptid_viscode, all_labels, mean, std, batch_size=1, dim=config.input_shape, n_channels=1, n_classes=3,
                                                              shuffle=False)
                        elif config.non_image_data == True:
                            dim_non_imaging_features = (config.non_image_input_shape, )
                            if config.only_non_imaging == False:
                                train_generator = DataGeneratorMultipleInputs_random_scans(train_validation_ptid_viscode["train"][i], partition_train_validation_unique["train"][i], labels_ptid_viscode, all_labels, train_features, mean, std, non_image_mean, non_image_std,
                                                                              batch_size=config.batch_size, dim=config.input_shape, dim_non_image=dim_non_imaging_features,
                                                                              n_channels=1, n_classes=3, shuffle=True)
                                validation_generator = DataGeneratorMultipleInputs_random_scans(train_validation_ptid_viscode["validation"][i], partition_train_validation_unique["validation"][i], labels_ptid_viscode, all_labels, validation_features, mean, std,
                                                                                   non_image_mean, non_image_std, batch_size=config.batch_size, dim=config.input_shape,
                                                                                   dim_non_image=dim_non_imaging_features, n_channels=1, n_classes=3, shuffle=True)
                                CM_train_generator = DataGeneratorMultipleInputs_random_scans(train_validation_ptid_viscode["train"][i], partition_train_validation_unique["train"][i], labels_ptid_viscode, all_labels, train_features, mean, std, non_image_mean,
                                                                                 non_image_std, batch_size=1, dim=config.input_shape, dim_non_image=dim_non_imaging_features,
                                                                                 n_channels=1, n_classes=3, shuffle=False)
                                CM_validation_generator = DataGeneratorMultipleInputs_random_scans(train_validation_ptid_viscode["validation"][i], partition_train_validation_unique["validation"][i], labels_ptid_viscode, all_labels, validation_features, mean, std,
                                                                                      non_image_mean, non_image_std, batch_size=1, dim=config.input_shape,
                                                                                      dim_non_image=dim_non_imaging_features, n_channels=1, n_classes=3, shuffle=False)
                                CM_test_generator = DataGeneratorMultipleInputs_random_scans(train_test_ptid_viscode["test"][i], partition_train_test_unique["test"][i], labels_ptid_viscode, all_labels, test_features, mean, std, non_image_mean,
                                                                                non_image_std, batch_size=1, dim=config.input_shape,
                                                                                dim_non_image=dim_non_imaging_features, n_channels=1, n_classes=3, shuffle=False)
                            else:  # if config.only_non_imaging == True
                                train_generator = DataGeneratorOnlyNonImagingFeatures_random_scans(train_validation_ptid_viscode["train"][i], partition_train_validation_unique["train"][i], labels_ptid_viscode, all_labels, train_features, non_image_mean,
                                                                                      non_image_std, batch_size=config.batch_size, dim_non_image=dim_non_imaging_features,
                                                                                      n_channels=1, n_classes=3, shuffle=True)
                                validation_generator = DataGeneratorOnlyNonImagingFeatures_random_scans(train_validation_ptid_viscode["validation"][i], partition_train_validation_unique["validation"][i], labels_ptid_viscode, all_labels, validation_features, non_image_mean,
                                                                                        non_image_std, batch_size=config.batch_size, dim_non_image=dim_non_imaging_features,
                                                                                        n_channels=1, n_classes=3, shuffle=True)
                                CM_train_generator = DataGeneratorOnlyNonImagingFeatures_random_scans(train_validation_ptid_viscode["train"][i], partition_train_validation_unique["train"][i], labels_ptid_viscode, all_labels, train_features, non_image_mean,
                                                                                         non_image_std, batch_size=1, dim_non_image=dim_non_imaging_features,
                                                                                         n_channels=1, n_classes=3, shuffle=False)
                                CM_validation_generator = DataGeneratorOnlyNonImagingFeatures_random_scans(train_validation_ptid_viscode["validation"][i], partition_train_validation_unique["validation"][i], labels_ptid_viscode, all_labels, validation_features, non_image_mean,
                                                                                        non_image_std, batch_size=1, dim_non_image=dim_non_imaging_features,
                                                                                        n_channels=1, n_classes=3, shuffle=False)
                                CM_test_generator = DataGeneratorOnlyNonImagingFeatures_random_scans(train_test_ptid_viscode["test"][i], partition_train_test_unique["test"][i], labels_ptid_viscode, all_labels, test_features, non_image_mean,
                                                                                        non_image_std, batch_size=1, dim_non_image=dim_non_imaging_features,
                                                                                        n_channels=1, n_classes=3, shuffle=False)

                        # set callbacks
                        callback_list = callbacks_list(CM_train_generator, CM_validation_generator, labels_ptid_viscode, results_dir)

                        if not config.fully_trained:

                            # TRAINING
                            history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                                          class_weight=None, callbacks=callback_list, epochs=config.epochs,
                                                          verbose=1, use_multiprocessing=False, workers=1, shuffle=False)

                        count_retrain_patience += 1

                if not config.fully_trained:
                    # plot acc + loss
                    plot_acc_loss(history, results_dir, i, callback_list[0], callback_list[1])

                    # plot performance per epoch
                    if config.epoch_performance:
                        plot_epoch_performance(callback_list[0])  # training performance
                        plot_epoch_performance(callback_list[1])  # validation performance

                    # load model of epoch with best performance
                    model = load_best_model(results_dir)

                    # TRAINING EVALUATION
                    print("\n---------------------- TRAINING EVALUATION ----------------------\n\n")

                    # mauc, bca
                    Y_pred = model.predict_generator(CM_train_generator, verbose=0)
                    Y_pred_converters = []
                    Y_pred_non_converters = []
                    y_true = []
                    y_true_converters = []
                    y_true_non_converters = []
                    count = 0
                    for id in CM_train_generator.list_IDs:
                        y_true.append(labels_ptid_viscode[id])
                        all_links = all_labels[all_labels["PTID_VISCODE"] == id]
                        this_link = all_links[all_links["Link"].isin(partition_train_validation_unique["train"][i])]
                        index = this_link.index.item()
                        if this_link.loc[index, "Converter"] == True:
                            y_true_converters.append(labels_ptid_viscode[id])
                            Y_pred_converters.append(Y_pred[count, :])
                        else:
                            y_true_non_converters.append(labels_ptid_viscode[id])
                            Y_pred_non_converters.append(Y_pred[count, :])
                        count += 1

                    Y_pred_converters = np.asarray(Y_pred_converters)
                    Y_pred_non_converters = np.asarray(Y_pred_non_converters)

                    if config.ordered_softmax == True:
                        ordinal_to_prob = np.zeros((Y_pred.shape[0], Y_pred.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred[:, 0]) - Y_pred[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred[:, 0]) - np.square(Y_pred[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred[:, 0]) - np.square(1 - Y_pred[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred = sftmx(ordinal_to_prob, axis=1)

                        ordinal_to_prob = np.zeros((Y_pred_converters.shape[0], Y_pred_converters.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred_converters[:, 0]) - Y_pred_converters[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred_converters[:, 0]) - np.square(Y_pred_converters[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred_converters[:, 0]) - np.square(1 - Y_pred_converters[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred_converters = sftmx(ordinal_to_prob, axis=1)

                        ordinal_to_prob = np.zeros((Y_pred_non_converters.shape[0], Y_pred_non_converters.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred_non_converters[:, 0]) - Y_pred_non_converters[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred_non_converters[:, 0]) - np.square(Y_pred_non_converters[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred_non_converters[:, 0]) - np.square(1 - Y_pred_non_converters[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred_non_converters = sftmx(ordinal_to_prob, axis=1)

                    y_pred = np.argmax(Y_pred, axis=1)
                    y_pred_converters = np.argmax(Y_pred_converters, axis=1)
                    y_pred_non_converters = np.argmax(Y_pred_non_converters, axis=1)


                    # calculate MAUC based on prediction estimates
                    nrSubj = Y_pred.shape[0]
                    nrClasses = Y_pred.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred[s, 0]
                        pMCI = Y_pred[s, 1]
                        pAD = Y_pred[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true[s], [pCN, pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    mauc = MAUC(zipTrueLabelAndProbs, nrClasses)

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true)
                    bca = calcBCA(hardEstimClass, true_labels, nrClasses)
                    conf = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix TRAINING set:\n")
                    file.write(np.array2string(conf, separator=', '))
                    print(conf)
                    acc = np.trace(conf) / float(np.sum(conf))
                    ari = adjusted_rand_score(hardEstimClass, true_labels)

                    # calculate MAUC based on prediction estimates ONLY FOR CONVERTERS
                    nrSubj = Y_pred_converters.shape[0]
                    nrClasses = Y_pred_converters.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred_converters[s, 0]
                        pMCI = Y_pred_converters[s, 1]
                        pAD = Y_pred_converters[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true_converters[s] - 1, [pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    try:
                        auc_converters = MAUC(zipTrueLabelAndProbs, nrClasses - 1)
                    except:
                        (print("MAUC could not be calculated"))
                        auc_converters = 0

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true_converters)
                    try:
                        bca_converters = calcBCA(hardEstimClass, true_labels, nrClasses)
                    except:
                        (print("BCA could not be calculated"))
                        bca_converters = 0
                    conf_converters = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix TRAINING set CONVERTERS:\n")
                    file.write(np.array2string(conf_converters, separator=', '))
                    print(conf_converters)
                    acc_converters = np.trace(conf_converters) / float(np.sum(conf_converters))
                    try:
                        ari_converters = adjusted_rand_score(hardEstimClass, true_labels)
                    except:
                        (print("ARI could not be calculated"))
                        ari_converters = 0

                    # calculate MAUC based on prediction estimates ONLY FOR NON CONVERTERS
                    nrSubj = Y_pred_non_converters.shape[0]
                    nrClasses = Y_pred_non_converters.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred_non_converters[s, 0]
                        pMCI = Y_pred_non_converters[s, 1]
                        pAD = Y_pred_non_converters[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true_non_converters[s], [pCN, pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    mauc_non_converters = MAUC(zipTrueLabelAndProbs, nrClasses)

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true_non_converters)
                    bca_non_converters = calcBCA(hardEstimClass, true_labels, nrClasses)
                    conf_non_converters = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix TRAINING set NON CONVERTERS:\n")
                    file.write(np.array2string(conf_non_converters, separator=', '))
                    print(conf_non_converters)
                    acc_non_converters = np.trace(conf_non_converters) / float(np.sum(conf_non_converters))
                    ari_non_converters = adjusted_rand_score(hardEstimClass, true_labels)


                    # loss, acc
                    if config.loss_function != OCC.customLossFunction:
                        score = model.evaluate_generator(generator=train_generator, verbose=1)
                        results["train"]["loss"].append(score[0])
                    # score_loss_in_CM_train_generator = model.evaluate_generator(generator=CM_train_generator, verbose=1)

                    results["train"]["mauc"].append(mauc)
                    results["train"]["bca"].append(bca)
                    results["train"]["acc"].append(acc)
                    results["train"]["ari"].append(ari)
                    results["train"]["auc_converters"].append(auc_converters)
                    results["train"]["bca_converters"].append(bca_converters)
                    results["train"]["acc_converters"].append(acc_converters)
                    results["train"]["ari_converters"].append(ari_converters)
                    results["train"]["mauc_non_converters"].append(mauc_non_converters)
                    results["train"]["bca_non_converters"].append(bca_non_converters)
                    results["train"]["acc_non_converters"].append(acc_non_converters)
                    results["train"]["ari_non_converters"].append(ari_non_converters)

                    # save classification per subject (for statistical test)
                    np.save(results_dir + "/train_IDs.npy", CM_train_generator.list_IDs)
                    np.save(results_dir + "/train_y_true.npy", y_true)
                    np.save(results_dir + "/train_y_pred.npy", y_pred)
                    np.save(results_dir + "/train_y_true_converters.npy", y_true_converters)
                    np.save(results_dir + "/train_y_pred_converters.npy", y_pred_converters)
                    np.save(results_dir + "/train_y_true_non_converters.npy", y_true_non_converters)
                    np.save(results_dir + "/train_y_pred_non_converters.npy", y_pred_non_converters)

                    # report train results
                    if config.loss_function != OCC.customLossFunction:
                        train_results = f"\nTrain\n    loss: {score[0]:.4f}\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    ACC: {acc:.4f}\n    ARI: {ari:.4f}\n    AUC CONVERTERS: {auc_converters:.4f}\n    BCA CONVERTERS: {bca_converters:.4f}\n    ACC CONVERTERS: {acc_converters:.4f}\n    ARI CONVERTERS: {ari_converters:.4f}\n    MAUC NON CONVERTERS: {mauc_non_converters:.4f}\n    BCA NON CONVERTERS: {bca_non_converters:.4f}\n    ACC NON CONVERTERS: {acc_non_converters:.4f}\n    ARI NON CONVERTERS: {ari_non_converters:.4f}\n    "
                    else:
                        train_results = f"\nTrain\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    ACC: {acc:.4f}\n    ARI: {ari:.4f}\n    AUC CONVERTERS: {auc_converters:.4f}\n    BCA CONVERTERS: {bca_converters:.4f}\n    ACC CONVERTERS: {acc_converters:.4f}\n    ARI CONVERTERS: {ari_converters:.4f}\n    MAUC NON CONVERTERS: {mauc_non_converters:.4f}\n    BCA NON CONVERTERS: {bca_non_converters:.4f}\n    ACC NON CONVERTERS: {acc_non_converters:.4f}\n    ARI NON CONVERTERS: {ari_non_converters:.4f}\n    "
                    file.write(train_results), print(train_results)

                    # VALIDATION EVALUATION
                    print("\n---------------------- VALIDATION EVALUATION ----------------------\n\n")

                    # mauc, bca
                    Y_pred = model.predict_generator(CM_validation_generator, verbose=0)
                    Y_pred_converters = []
                    Y_pred_non_converters = []
                    y_true = []
                    y_true_converters = []
                    y_true_non_converters = []
                    count = 0
                    for id in CM_validation_generator.list_IDs:
                        y_true.append(labels_ptid_viscode[id])
                        all_links = all_labels[all_labels["PTID_VISCODE"] == id]
                        this_link = all_links[all_links["Link"].isin(partition_train_validation_unique["validation"][i])]
                        index = this_link.index.item()
                        if this_link.loc[index, "Converter"] == True:
                            y_true_converters.append(labels_ptid_viscode[id])
                            Y_pred_converters.append(Y_pred[count, :])
                        else:
                            y_true_non_converters.append(labels_ptid_viscode[id])
                            Y_pred_non_converters.append(Y_pred[count, :])
                        count += 1

                    Y_pred_converters = np.asarray(Y_pred_converters)
                    Y_pred_non_converters = np.asarray(Y_pred_non_converters)

                    if config.ordered_softmax == True:
                        ordinal_to_prob = np.zeros((Y_pred.shape[0], Y_pred.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred[:, 0]) - Y_pred[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred[:, 0]) - np.square(Y_pred[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred[:, 0]) - np.square(1 - Y_pred[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred = sftmx(ordinal_to_prob, axis=1)

                        ordinal_to_prob = np.zeros((Y_pred_converters.shape[0], Y_pred_converters.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred_converters[:, 0]) - Y_pred_converters[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred_converters[:, 0]) - np.square(Y_pred_converters[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred_converters[:, 0]) - np.square(1 - Y_pred_converters[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred_converters = sftmx(ordinal_to_prob, axis=1)

                        ordinal_to_prob = np.zeros((Y_pred_non_converters.shape[0], Y_pred_non_converters.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred_non_converters[:, 0]) - Y_pred_non_converters[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred_non_converters[:, 0]) - np.square(Y_pred_non_converters[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred_non_converters[:, 0]) - np.square(1 - Y_pred_non_converters[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred_non_converters = sftmx(ordinal_to_prob, axis=1)

                    y_pred = np.argmax(Y_pred, axis=1)
                    y_pred_converters = np.argmax(Y_pred_converters, axis=1)
                    y_pred_non_converters = np.argmax(Y_pred_non_converters, axis=1)


                    # calculate MAUC based on prediction estimates
                    nrSubj = Y_pred.shape[0]
                    nrClasses = Y_pred.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred[s, 0]
                        pMCI = Y_pred[s, 1]
                        pAD = Y_pred[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true[s], [pCN, pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    mauc = MAUC(zipTrueLabelAndProbs, nrClasses)

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true)
                    bca = calcBCA(hardEstimClass, true_labels, nrClasses)
                    conf = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix VALIDATION set:\n")
                    file.write(np.array2string(conf, separator=', '))
                    print(conf)
                    acc = np.trace(conf) / float(np.sum(conf))
                    ari = adjusted_rand_score(hardEstimClass, true_labels)

                    # calculate MAUC based on prediction estimates ONLY FOR CONVERTERS
                    nrSubj = Y_pred_converters.shape[0]
                    nrClasses = Y_pred_converters.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred_converters[s, 0]
                        pMCI = Y_pred_converters[s, 1]
                        pAD = Y_pred_converters[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true_converters[s] - 1, [pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    try:
                        auc_converters = MAUC(zipTrueLabelAndProbs, nrClasses - 1)
                    except:
                        (print("MAUC could not be calculated"))
                        auc_converters = 0

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true_converters)
                    try:
                        bca_converters = calcBCA(hardEstimClass, true_labels, nrClasses)
                    except:
                        (print("BCA could not be calculated"))
                        bca_converters = 0
                    conf_converters = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix VALIDATION set CONVERTERS:\n")
                    file.write(np.array2string(conf_converters, separator=', '))
                    print(conf_converters)
                    acc_converters = np.trace(conf_converters) / float(np.sum(conf_converters))
                    try:
                        ari_converters = adjusted_rand_score(hardEstimClass, true_labels)
                    except:
                        (print("ARI could not be calculated"))
                        ari_converters = 0

                    # calculate MAUC based on prediction estimates ONLY FOR NON CONVERTERS
                    nrSubj = Y_pred_non_converters.shape[0]
                    nrClasses = Y_pred_non_converters.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred_non_converters[s, 0]
                        pMCI = Y_pred_non_converters[s, 1]
                        pAD = Y_pred_non_converters[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true_non_converters[s], [pCN, pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    mauc_non_converters = MAUC(zipTrueLabelAndProbs, nrClasses)

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true_non_converters)
                    bca_non_converters = calcBCA(hardEstimClass, true_labels, nrClasses)
                    conf_non_converters = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix VALIDATION set NON CONVERTERS:\n")
                    file.write(np.array2string(conf_non_converters, separator=', '))
                    print(conf_non_converters)
                    acc_non_converters = np.trace(conf_non_converters) / float(np.sum(conf_non_converters))
                    ari_non_converters = adjusted_rand_score(hardEstimClass, true_labels)


                    # loss, acc
                    if config.loss_function != OCC.customLossFunction:
                        score = model.evaluate_generator(generator=validation_generator, verbose=1)
                        results["validation"]["loss"].append(score[0])

                    results["validation"]["mauc"].append(mauc)
                    results["validation"]["bca"].append(bca)
                    results["validation"]["acc"].append(acc)
                    results["validation"]["ari"].append(ari)
                    results["validation"]["auc_converters"].append(auc_converters)
                    results["validation"]["bca_converters"].append(bca_converters)
                    results["validation"]["acc_converters"].append(acc_converters)
                    results["validation"]["ari_converters"].append(ari_converters)
                    results["validation"]["mauc_non_converters"].append(mauc_non_converters)
                    results["validation"]["bca_non_converters"].append(bca_non_converters)
                    results["validation"]["acc_non_converters"].append(acc_non_converters)
                    results["validation"]["ari_non_converters"].append(ari_non_converters)

                    # save classification per subject (for statistical test)
                    np.save(results_dir + "/val_IDs.npy", CM_validation_generator.list_IDs)
                    np.save(results_dir + "/val_y_true.npy", y_true)
                    np.save(results_dir + "/val_y_pred.npy", y_pred)
                    np.save(results_dir + "/val_y_true_converters.npy", y_true_converters)
                    np.save(results_dir + "/val_y_pred_converters.npy", y_pred_converters)
                    np.save(results_dir + "/val_y_true_non_converters.npy", y_true_non_converters)
                    np.save(results_dir + "/val_y_pred_non_converters.npy", y_pred_non_converters)

                    # report val results
                    if config.loss_function != OCC.customLossFunction:
                        val_results = f"\nValidation\n    loss: {score[0]:.4f}\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    ACC: {acc:.4f}\n    ARI: {ari:.4f}\n    AUC CONVERTERS: {auc_converters:.4f}\n    BCA CONVERTERS: {bca_converters:.4f}\n    ACC CONVERTERS: {acc_converters:.4f}\n    ARI CONVERTERS: {ari_converters:.4f}\n    MAUC NON CONVERTERS: {mauc_non_converters:.4f}\n    BCA NON CONVERTERS: {bca_non_converters:.4f}\n    ACC NON CONVERTERS: {acc_non_converters:.4f}\n    ARI NON CONVERTERS: {ari_non_converters:.4f}\n    "

                    else:
                        val_results = f"\nValidation\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    ACC: {acc:.4f}\n    ARI: {ari:.4f}\n    AUC CONVERTERS: {auc_converters:.4f}\n    BCA CONVERTERS: {bca_converters:.4f}\n    ACC CONVERTERS: {acc_converters:.4f}\n    ARI CONVERTERS: {ari_converters:.4f}\n    MAUC NON CONVERTERS: {mauc_non_converters:.4f}\n    BCA NON CONVERTERS: {bca_non_converters:.4f}\n    ACC NON CONVERTERS: {acc_non_converters:.4f}\n    ARI NON CONVERTERS: {ari_non_converters:.4f}\n    "
                    file.write(val_results), print(val_results)


                # TESTING EVALUATION
                print("\n---------------------- TESTING EVALUATION ----------------------\n\n")

                # make predictions for test set
                Y_pred = model.predict_generator(CM_test_generator, verbose=0)

                if config.leaderboard:
                    # mauc and bca
                    Y_pred_converters = []
                    Y_pred_non_converters = []
                    y_true = []
                    y_true_converters = []
                    y_true_non_converters = []
                    count = 0
                    for id in CM_test_generator.list_IDs:
                        y_true.append(labels_ptid_viscode[id])
                        all_links = all_labels[all_labels["PTID_VISCODE"] == id]
                        this_link = all_links[all_links["Link"].isin(partition_train_test_unique["test"][i])]
                        index = this_link.index.item()
                        if this_link.loc[index, "Converter"] == True:
                            y_true_converters.append(labels_ptid_viscode[id])
                            Y_pred_converters.append(Y_pred[count, :])
                        else:
                            y_true_non_converters.append(labels_ptid_viscode[id])
                            Y_pred_non_converters.append(Y_pred[count, :])
                        count += 1

                    Y_pred_converters = np.asarray(Y_pred_converters)
                    Y_pred_non_converters = np.asarray(Y_pred_non_converters)

                    if config.ordered_softmax == True:
                        ordinal_to_prob = np.zeros((Y_pred.shape[0], Y_pred.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred[:, 0]) - Y_pred[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred[:, 0]) - np.square(Y_pred[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred[:, 0]) - np.square(1 - Y_pred[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred = sftmx(ordinal_to_prob, axis=1)

                        ordinal_to_prob = np.zeros((Y_pred_converters.shape[0], Y_pred_converters.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred_converters[:, 0]) - Y_pred_converters[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred_converters[:, 0]) - np.square(Y_pred_converters[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred_converters[:, 0]) - np.square(1 - Y_pred_converters[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred_converters = sftmx(ordinal_to_prob, axis=1)

                        ordinal_to_prob = np.zeros((Y_pred_non_converters.shape[0], Y_pred_non_converters.shape[1] + 1))
                        ordinal_to_prob[:, 0] = 1 - np.square(Y_pred_non_converters[:, 0]) - Y_pred_non_converters[:, 1]
                        ordinal_to_prob[:, 1] = 1 - np.square(1 - Y_pred_non_converters[:, 0]) - np.square(Y_pred_non_converters[:, 1])
                        ordinal_to_prob[:, 2] = 1 - (1 - Y_pred_non_converters[:, 0]) - np.square(1 - Y_pred_non_converters[:, 1])
                        ordinal_to_prob = ordinal_to_prob.clip(min=0)
                        Y_pred_non_converters = sftmx(ordinal_to_prob, axis=1)

                    y_pred = np.argmax(Y_pred, axis=1)
                    y_pred_converters = np.argmax(Y_pred_converters, axis=1)
                    y_pred_non_converters = np.argmax(Y_pred_non_converters, axis=1)


                    # calculate MAUC based on prediction estimates
                    nrSubj = Y_pred.shape[0]
                    nrClasses = Y_pred.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred[s, 0]
                        pMCI = Y_pred[s, 1]
                        pAD = Y_pred[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true[s], [pCN, pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    mauc = MAUC(zipTrueLabelAndProbs, nrClasses)

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true)
                    bca = calcBCA(hardEstimClass, true_labels, nrClasses)
                    conf = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix TEST set:\n")
                    file.write(np.array2string(conf, separator=', '))
                    print(conf)
                    acc = np.trace(conf) / float(np.sum(conf))
                    ari = adjusted_rand_score(hardEstimClass, true_labels)

                    # calculate MAUC based on prediction estimates ONLY FOR CONVERTERS
                    nrSubj = Y_pred_converters.shape[0]
                    nrClasses = Y_pred_converters.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred_converters[s, 0]
                        pMCI = Y_pred_converters[s, 1]
                        pAD = Y_pred_converters[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true_converters[s] - 1, [pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    try:
                        auc_converters = MAUC(zipTrueLabelAndProbs, nrClasses - 1)
                    except:
                        (print("MAUC could not be calculated"))
                        auc_converters = 0

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true_converters)
                    try:
                        bca_converters = calcBCA(hardEstimClass, true_labels, nrClasses)
                    except:
                        (print("BCA could not be calculated"))
                        bca_converters = 0
                    conf_converters = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix TEST set CONVERTERS:\n")
                    file.write(np.array2string(conf_converters, separator=', '))
                    print(conf_converters)
                    acc_converters = np.trace(conf_converters) / float(np.sum(conf_converters))
                    try:
                        ari_converters = adjusted_rand_score(hardEstimClass, true_labels)
                    except:
                        (print("ARI could not be calculated"))
                        ari_converters = 0

                    # calculate MAUC based on prediction estimates ONLY FOR NON CONVERTERS
                    nrSubj = Y_pred_non_converters.shape[0]
                    nrClasses = Y_pred_non_converters.shape[1]
                    hardEstimClass = -1 * np.ones(nrSubj, int)
                    zipTrueLabelAndProbs = []
                    for s in range(nrSubj):
                        pCN = Y_pred_non_converters[s, 0]
                        pMCI = Y_pred_non_converters[s, 1]
                        pAD = Y_pred_non_converters[s, 2]
                        hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
                        zipTrueLabelAndProbs += [(y_true_non_converters[s], [pCN, pMCI, pAD])]
                    zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
                    mauc_non_converters = MAUC(zipTrueLabelAndProbs, nrClasses)

                    # calculate BCA based on prediction estimates
                    true_labels = np.asarray(y_true_non_converters)
                    bca_non_converters = calcBCA(hardEstimClass, true_labels, nrClasses)
                    conf_non_converters = confusion_matrix(hardEstimClass, true_labels, [0, 1, 2])
                    file.write("Confusion matrix TEST set NON CONVERTERS:\n")
                    file.write(np.array2string(conf_non_converters, separator=', '))
                    print(conf_non_converters)
                    acc_non_converters = np.trace(conf_non_converters) / float(np.sum(conf_non_converters))
                    ari_non_converters = adjusted_rand_score(hardEstimClass, true_labels)


                    # loss, acc
                    # score = model.evaluate_generator(generator=test_generator, verbose=1)

                    # results["test"]["loss"].append(score[0])
                    results["test"]["mauc"].append(mauc)
                    results["test"]["bca"].append(bca)
                    results["test"]["acc"].append(acc)
                    results["test"]["ari"].append(ari)
                    results["test"]["auc_converters"].append(auc_converters)
                    results["test"]["bca_converters"].append(bca_converters)
                    results["test"]["acc_converters"].append(acc_converters)
                    results["test"]["ari_converters"].append(ari_converters)
                    results["test"]["mauc_non_converters"].append(mauc_non_converters)
                    results["test"]["bca_non_converters"].append(bca_non_converters)
                    results["test"]["acc_non_converters"].append(acc_non_converters)
                    results["test"]["ari_non_converters"].append(ari_non_converters)

                    # save classification per subject (for statistical test)
                    np.save(results_dir + "/test_IDs.npy", CM_test_generator.list_IDs)
                    np.save(results_dir + "/test_y_true.npy", y_true)
                    np.save(results_dir + "/test_y_pred.npy", y_pred)
                    np.save(results_dir + "/test_y_true_converters.npy", y_true_converters)
                    np.save(results_dir + "/test_y_pred_converters.npy", y_pred_converters)
                    np.save(results_dir + "/test_y_true_non_converters.npy", y_true_non_converters)
                    np.save(results_dir + "/test_y_pred_non_converters.npy", y_pred_non_converters)

                    # report test results
                    test_results = f"\nTest\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    ACC: {acc:.4f}\n    ARI: {ari:.4f}\n    AUC CONVERTERS: {auc_converters:.4f}\n    BCA CONVERTERS: {bca_converters:.4f}\n    ACC CONVERTERS: {acc_converters:.4f}\n    ARI CONVERTERS: {ari_converters:.4f}\n    MAUC NON CONVERTERS: {mauc_non_converters:.4f}\n    BCA NON CONVERTERS: {bca_non_converters:.4f}\n    ACC NON CONVERTERS: {acc_non_converters:.4f}\n    ARI NON CONVERTERS: {ari_non_converters:.4f}\n    "
                    file.write(test_results), print(test_results)

                file.close()

                # delete augmented images
                if config.augmentation:
                    #os.system('rm -rf %s/*' % config.aug_dir)
                    shutil.rmtree(config.aug_dir)


            # end timer
            end = time.time()
            end_localtime = time.localtime()

            # save results + model
            np.save(config.output_dir + "results.npy", results)
            # save_DL_model(model)
            save_results(results, start, start_localtime, end, end_localtime)

            # Add Test results summary
            test_results = {}
            # test_results['acc_mean'] = np.mean(results['test']['acc'])
            # test_results['acc_95ci'] = compute_CI.compute_confidence(results['test']['acc'], N_1, N_2, alpha)
            # test_results['auc_mean'] = np.mean(results['test']['auc'])
            # test_results['auc_95ci'] = compute_CI.compute_confidence(results['test']['auc'], N_1, N_2, alpha)

            # ALL
            test_results['mauc_mean'] = np.mean(results['test']['mauc'])
            test_results['mauc_std'] = np.std(results['test']['mauc'])
            #test_results['mauc_95ci'] = compute_CI.compute_confidence(results['test']['mauc'], N_1, N_2, alpha) # change N !
            # if test_results['mauc_95ci'][0] < 0:
            #     test_results['mauc_95ci'][0] = 0
            # if test_results['mauc_95ci'][1] > 1:
            #     test_results['mauc_95ci'][1] = 1
            test_results['bca_mean'] = np.mean(results['test']['bca'])
            test_results['bca_std'] = np.std(results['test']['bca'])
            #test_results['bca_95ci'] = compute_CI.compute_confidence(results['test']['bca'], N_1, N_2, alpha) # change N !
            # if test_results['bca_95ci'][0] < 0:
            #     test_results['bca_95ci'][0] = 0
            # if test_results['bca_95ci'][1] > 1:
            #     test_results['bca_95ci'][1] = 1
            test_results['acc_mean'] = np.mean(results['test']['acc'])
            test_results['acc_std'] = np.std(results['test']['acc'])
            # test_results['acc_95ci'] = compute_CI.compute_confidence(results['test']['acc'], N_1, N_2, alpha) # change N !
            test_results['ari_mean'] = np.mean(results['test']['ari'])
            test_results['ari_std'] = np.std(results['test']['ari'])

            # CONVERTERS
            test_results['auc_mean_converters'] = np.mean(results['test']['auc_converters'])
            test_results['auc_std_converters'] = np.std(results['test']['auc_converters'])
            #test_results['mauc_95ci_converters'] = compute_CI.compute_confidence(results['test']['auc_converters'], N_1_converters, N_2_converters, alpha) # change N !
            # if test_results['mauc_95ci'][0] < 0:
            #     test_results['mauc_95ci'][0] = 0
            # if test_results['mauc_95ci'][1] > 1:
            #     test_results['mauc_95ci'][1] = 1
            test_results['bca_mean_converters'] = np.mean(results['test']['bca_converters'])
            test_results['bca_std_converters'] = np.std(results['test']['bca_converters'])
            #test_results['bca_95ci_converters'] = compute_CI.compute_confidence(results['test']['bca_converters'], N_1_converters, N_2_converters, alpha) # change N !
            # if test_results['bca_95ci'][0] < 0:
            #     test_results['bca_95ci'][0] = 0
            # if test_results['bca_95ci'][1] > 1:
            #     test_results['bca_95ci'][1] = 1
            test_results['acc_mean_converters'] = np.mean(results['test']['acc_converters'])
            test_results['acc_std_converters'] = np.std(results['test']['acc_converters'])
            # test_results['bca_95ci_converters'] = compute_CI.compute_confidence(results['test']['bca_converters'], N_1_converters, N_2_converters, alpha) # change N !
            test_results['ari_mean_converters'] = np.mean(results['test']['ari_converters'])
            test_results['ari_std_converters'] = np.std(results['test']['ari_converters'])

            # NON CONVERTERS
            test_results['mauc_mean_non_converters'] = np.mean(results['test']['mauc_non_converters'])
            test_results['mauc_std_non_converters'] = np.std(results['test']['mauc_non_converters'])
            #test_results['mauc_95ci_non_converters'] = compute_CI.compute_confidence(results['test']['mauc_non_converters'], N_1_non_converters, N_2_non_converters, alpha) # change N !
            # if test_results['mauc_95ci'][0] < 0:
            #     test_results['mauc_95ci'][0] = 0
            # if test_results['mauc_95ci'][1] > 1:
            #     test_results['mauc_95ci'][1] = 1
            test_results['bca_mean_non_converters'] = np.mean(results['test']['bca_non_converters'])
            test_results['bca_std_non_converters'] = np.std(results['test']['bca_non_converters'])
            #test_results['bca_95ci_non_converters'] = compute_CI.compute_confidence(results['test']['bca_non_converters'], N_1_non_converters, N_2_non_converters, alpha) # change N !
            # if test_results['bca_95ci'][0] < 0:
            #     test_results['bca_95ci'][0] = 0
            # if test_results['bca_95ci'][1] > 1:
            #     test_results['bca_95ci'][1] = 1
            test_results['acc_mean_non_converters'] = np.mean(results['test']['acc_non_converters'])
            test_results['acc_std_non_converters'] = np.std(results['test']['acc_non_converters'])
            # test_results['bca_95ci_non_converters'] = compute_CI.compute_confidence(results['test']['bca_non_converters'], N_1_non_converters, N_2_non_converters, alpha) # change N !
            test_results['ari_mean_non_converters'] = np.mean(results['test']['ari_non_converters'])
            test_results['ari_std_non_converters'] = np.std(results['test']['ari_non_converters'])

            # print('mean mauc:', '%.3f' % test_results['mauc_mean'], ' (', '%.3f' % test_results['mauc_95ci'][0], ' - ',
            #      '%.3f' % test_results['mauc_95ci'][1], ')')
            # print('mean bca:', '%.3f' % test_results['bca_mean'], ' (', '%.3f' % test_results['bca_95ci'][0], ' - ',
            #      '%.3f' % test_results['bca_95ci'][1], ')')
            print('mean mauc:', '%.3f' % test_results['mauc_mean'], ' (', '%.3f' % test_results['mauc_std'], ')')
            print('mean bca:', '%.3f' % test_results['bca_mean'], ' (', '%.3f' % test_results['bca_std'], ')')
            print('mean acc:', '%.3f' % test_results['acc_mean'], ' (', '%.3f' % test_results['acc_std'], ')')
            print('mean ari:', '%.3f' % test_results['ari_mean'], ' (', '%.3f' % test_results['ari_std'], ')')
            # print('mean mauc converters:', '%.3f' % test_results['mauc_mean_converters'])
            print('mean auc converters:', '%.3f' % test_results['auc_mean_converters'], ' (', '%.3f' % test_results['auc_std_converters'], ')')
            #print('mean mauc converters:', '%.3f' % test_results['mauc_mean_converters'], ' (',
            #      '%.3f' % test_results['mauc_95ci_converters'][0], ' - ',
            #      '%.3f' % test_results['mauc_95ci_converters'][1], ')')
            # print('mean bca converters:', '%.3f' % test_results['bca_mean_converters'])
            print('mean bca converters:', '%.3f' % test_results['bca_mean_converters'], ' (', '%.3f' % test_results['bca_std_converters'], ')')
            #print('mean bca converters:', '%.3f' % test_results['bca_mean_converters'], ' (',
            #      '%.3f' % test_results['bca_95ci_converters'][0], ' - ',
            #      '%.3f' % test_results['bca_95ci_converters'][1], ')')
            print('mean acc converters:', '%.3f' % test_results['acc_mean_converters'], ' (', '%.3f' % test_results['acc_std_converters'], ')')
            print('mean ari converters:', '%.3f' % test_results['ari_mean_converters'], ' (', '%.3f' % test_results['ari_std_converters'], ')')
            # print('mean mauc non converters:', '%.3f' % test_results['mauc_mean_non_converters'])
            print('mean mauc non converters:', '%.3f' % test_results['mauc_mean_non_converters'], ' (', '%.3f' % test_results['mauc_std_non_converters'], ')')
            #print('mean mauc non converters:', '%.3f' % test_results['mauc_mean_non_converters'], ' (',
            #      '%.3f' % test_results['mauc_95ci_non_converters'][0], ' - ',
            #      '%.3f' % test_results['mauc_95ci_non_converters'][1], ')')
            # print('mean bca non converters:', '%.3f' % test_results['bca_mean_non_converters'])
            print('mean bca non converters:', '%.3f' % test_results['bca_mean_non_converters'], ' (', '%.3f' % test_results['bca_std_non_converters'], ')')
            #print('mean bca non converters:', '%.3f' % test_results['bca_mean_non_converters'], ' (',
            #      '%.3f' % test_results['bca_95ci_non_converters'][0], ' - ',
            #      '%.3f' % test_results['bca_95ci_non_converters'][1], ')')
            print('mean acc non converters:', '%.3f' % test_results['acc_mean_non_converters'], ' (', '%.3f' % test_results['acc_std_non_converters'], ')')
            print('mean ari non converters:', '%.3f' % test_results['ari_mean_non_converters'], ' (', '%.3f' % test_results['ari_std_non_converters'], ')')

            np.save(config.output_dir + "test_results.npy", test_results)


        print('\nend')




def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main(sys.argv)
