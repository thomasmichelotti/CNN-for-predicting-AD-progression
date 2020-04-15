import numpy as np
import pandas as pd
import os
from keras.engine.saving import load_model
#import C_MainScripts.ordinal_categorical_crossentropy as OCC
import time
from C_MainScripts.PSI_generators import PSI_DataGenerator, PSI_DataGenerator_multiple_inputs
from C_MainScripts.MAUC import MAUC
from C_MainScripts.evalOneSubmissionExtended import calcBCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from C_MainScripts.PSI_savings import save_results_PSI
import config_PSI
import sys



def main(argv):

    # start timer
    start = time.time()
    start_localtime = time.localtime()

    if len(argv) > 1:
        config_PSI.image_dir = sys.argv[1] + "/"

    # load data
    labels = pd.read_csv(config_PSI.data_dir + "labels.csv")
    non_image_features = pd.read_csv(config_PSI.data_dir + "all_non_image_features.csv")

    # create dict for labels
    labels_dict = dict()
    for index, row in labels.iterrows():
        if labels.loc[index, "Diagnosis"] == 1.0:
            labels_dict[labels.loc[index, "RID"]] = 0
        elif labels.loc[index, "Diagnosis"] == 2.0:
            labels_dict[labels.loc[index, "RID"]] = 1
        elif labels.loc[index, "Diagnosis"] == 3.0:
            labels_dict[labels.loc[index, "RID"]] = 2




    # initialization of results dictionary
    results = {"test": {"mauc": [], "bca": [], "acc": [], "ari": [], "auc_converters": [], "bca_converters": [], "acc_converters": [], "ari_converters": [], "mauc_non_converters": [], "bca_non_converters": [], "acc_non_converters": [], "ari_non_converters": []}}



    # get results for all 30 model iterations (different train-validation-test splits)
    for i in range(30):
        iteration = "k" + str(i)
        print(iteration)
        this_dir = config_PSI.new_dir + iteration

        L = os.listdir(this_dir)
        L.sort()
        model = load_model(this_dir + "/" + L[-1], compile=False)
        print("Loading best model for evaluation: ", L[-1])

        # create results dir for this iteration
        results_dir = config_PSI.output_dir + "k" + str(i)
        create_data_directory(results_dir)
        file = open(results_dir + "/results.txt", 'w')

        # get mean and std for this iteration
        mean = np.load(this_dir + "/mean.npy")
        std = np.load(this_dir + "/std.npy")
        if config_PSI.non_image_data == True:
            non_image_mean = np.load(this_dir + "/non_image_mean.npy")
            non_image_std = np.load(this_dir + "/non_image_std.npy")

        # create test generator
        if config_PSI.non_image_data == False:
            test_generator = PSI_DataGenerator(labels["RID"], labels_dict, mean, std, batch_size=1, dim=config_PSI.input_shape, n_channels=1, n_classes=3, shuffle=False)
        else:
            test_generator = PSI_DataGenerator_multiple_inputs(labels["RID"], labels_dict, non_image_features, mean, std, non_image_mean, non_image_std, batch_size=1,
                       dim=config_PSI.input_shape, dim_non_image=config_PSI.dim_non_imaging_features, n_channels=1, n_classes=3, shuffle=False)

        # make predictions for PSI
        Y_pred = model.predict_generator(test_generator, verbose=0)

        # mauc and bca
        Y_pred_converters = []
        Y_pred_non_converters = []
        y_true = []
        y_true_converters = []
        y_true_non_converters = []
        count = 0
        for id in test_generator.list_IDs:
            y_true.append(labels_dict[id])
            all_links = labels[labels["RID"] == id]
            this_link = all_links.iloc[[0]]
            index = this_link.index.item()
            if this_link.loc[index, "Converter"] == True:
                y_true_converters.append(labels_dict[id])
                Y_pred_converters.append(Y_pred[count, :])
            else:
                y_true_non_converters.append(labels_dict[id])
                Y_pred_non_converters.append(Y_pred[count, :])
            count += 1

        Y_pred_converters = np.asarray(Y_pred_converters)
        Y_pred_non_converters = np.asarray(Y_pred_non_converters)

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

        # report test results
        test_results = f"\nTest\n    MAUC: {mauc:.4f}\n    BCA: {bca:.4f}\n    ACC: {acc:.4f}\n    ARI: {ari:.4f}\n    AUC CONVERTERS: {auc_converters:.4f}\n    BCA CONVERTERS: {bca_converters:.4f}\n    ACC CONVERTERS: {acc_converters:.4f}\n    ARI CONVERTERS: {ari_converters:.4f}\n    MAUC NON CONVERTERS: {mauc_non_converters:.4f}\n    BCA NON CONVERTERS: {bca_non_converters:.4f}\n    ACC NON CONVERTERS: {acc_non_converters:.4f}\n    ARI NON CONVERTERS: {ari_non_converters:.4f}\n    "
        file.write(test_results), print(test_results)

        file.close()

    # end timer
    end = time.time()
    end_localtime = time.localtime()


    np.save(config_PSI.output_dir + "results.npy", results)
    save_results_PSI(results, start, start_localtime, end, end_localtime)

    test_results = {}
    test_results['mauc_mean'] = np.mean(results['test']['mauc'])
    test_results['mauc_std'] = np.std(results['test']['mauc'])
    test_results['bca_mean'] = np.mean(results['test']['bca'])
    test_results['bca_std'] = np.std(results['test']['bca'])
    test_results['acc_mean'] = np.mean(results['test']['acc'])
    test_results['acc_std'] = np.std(results['test']['acc'])
    test_results['ari_mean'] = np.mean(results['test']['ari'])
    test_results['ari_std'] = np.std(results['test']['ari'])

    # CONVERTERS
    test_results['auc_mean_converters'] = np.mean(results['test']['auc_converters'])
    test_results['auc_std_converters'] = np.std(results['test']['auc_converters'])
    test_results['bca_mean_converters'] = np.mean(results['test']['bca_converters'])
    test_results['bca_std_converters'] = np.std(results['test']['bca_converters'])
    test_results['acc_mean_converters'] = np.mean(results['test']['acc_converters'])
    test_results['acc_std_converters'] = np.std(results['test']['acc_converters'])
    test_results['ari_mean_converters'] = np.mean(results['test']['ari_converters'])
    test_results['ari_std_converters'] = np.std(results['test']['ari_converters'])

    # NON CONVERTERS
    test_results['mauc_mean_non_converters'] = np.mean(results['test']['mauc_non_converters'])
    test_results['mauc_std_non_converters'] = np.std(results['test']['mauc_non_converters'])
    test_results['bca_mean_non_converters'] = np.mean(results['test']['bca_non_converters'])
    test_results['bca_std_non_converters'] = np.std(results['test']['bca_non_converters'])
    test_results['acc_mean_non_converters'] = np.mean(results['test']['acc_non_converters'])
    test_results['acc_std_non_converters'] = np.std(results['test']['acc_non_converters'])
    test_results['ari_mean_non_converters'] = np.mean(results['test']['ari_non_converters'])
    test_results['ari_std_non_converters'] = np.std(results['test']['ari_non_converters'])

    print('mean mauc:', '%.3f' % test_results['mauc_mean'], ' (', '%.3f' % test_results['mauc_std'], ')')
    print('mean bca:', '%.3f' % test_results['bca_mean'], ' (', '%.3f' % test_results['bca_std'], ')')
    print('mean acc:', '%.3f' % test_results['acc_mean'], ' (', '%.3f' % test_results['acc_std'], ')')
    print('mean ari:', '%.3f' % test_results['ari_mean'], ' (', '%.3f' % test_results['ari_std'], ')')
    print('mean auc converters:', '%.3f' % test_results['auc_mean_converters'], ' (', '%.3f' % test_results['auc_std_converters'], ')')
    print('mean bca converters:', '%.3f' % test_results['bca_mean_converters'], ' (', '%.3f' % test_results['bca_std_converters'], ')')
    print('mean acc converters:', '%.3f' % test_results['acc_mean_converters'], ' (', '%.3f' % test_results['acc_std_converters'], ')')
    print('mean ari converters:', '%.3f' % test_results['ari_mean_converters'], ' (', '%.3f' % test_results['ari_std_converters'], ')')
    print('mean mauc non converters:', '%.3f' % test_results['mauc_mean_non_converters'], ' (', '%.3f' % test_results['mauc_std_non_converters'], ')')
    print('mean bca non converters:', '%.3f' % test_results['bca_mean_non_converters'], ' (', '%.3f' % test_results['bca_std_non_converters'], ')')
    print('mean acc non converters:', '%.3f' % test_results['acc_mean_non_converters'], ' (', '%.3f' % test_results['acc_std_non_converters'], ')')
    print('mean ari non converters:', '%.3f' % test_results['ari_mean_non_converters'], ' (', '%.3f' % test_results['ari_std_non_converters'], ')')

    np.save(config_PSI.output_dir + "test_results.npy", test_results)


def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main(sys.argv)


