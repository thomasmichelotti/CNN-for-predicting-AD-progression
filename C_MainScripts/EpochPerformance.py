import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, roc_auc_score
from C_MainScripts.MAUC import MAUC
from C_MainScripts.evalOneSubmissionExtended import calcBCA
from sklearn.metrics.cluster import adjusted_rand_score

import config
from scipy.special import softmax as sftmx

class EpochPerformance(Callback):
    """
    This callback class keeps track of the sensitivity, specificity, accuracy and AUC after each epoch.
    This can be used for both the training and validation data.

    The model with the best AUC score is saved, for later evaluation. When no AUC improvement after a
    specified amount of epochs early stopping is induced.
    """

    def __init__(self, generator, labels, name, results_dir):
        super(Callback).__init__()
        self.generator = generator
        self.labels = labels
        self.name = name
        self.results_dir = results_dir

    def on_train_begin(self, logs={}):
        self.auc_list = []
        self.sens_list = []
        self.spec_list = []
        self.acc_list = []
        self.acc_list_converters = []
        self.acc_list_non_converters = []
        self.best_epoch = 0
        self.mauc_list = []
        self.bca_list = []
        self.loss_list = []
        self.auc_list_converters = []
        self.bca_list_converters = []
        self.mauc_list_non_converters = []
        self.bca_list_non_converters = []
        self.ari_list = []
        self.ari_list_converters = []
        self.ari_list_non_converters = []

        return

    def on_epoch_end(self, epoch, logs={}):
        """
        After every epoch during training calculate the AUC, sens, spec and acc.
        Save model when AUC is improved and induce early stopping when necessary.
        """

        # split converters and non converters
        df = self.generator.df
        dataset_links = self.generator.list_IDs_link

        # get true labels and predicted labels
        Y_pred = self.model.predict_generator(self.generator, verbose=0)
        Y_pred_converters = []
        Y_pred_non_converters = []
        y_true = []
        y_true_converters = []
        y_true_non_converters = []
        count = 0
        for id in self.generator.list_IDs:
            y_true.append(self.labels[id])
            if id[0] != 'a':
                all_links = df[df["PTID_VISCODE"] == id]
                this_link = all_links[all_links["Link"].isin(dataset_links)]
                index = this_link.index.item()
                if this_link.loc[index, "Converter"] == True:
                    y_true_converters.append(self.labels[id])
                    Y_pred_converters.append(Y_pred[count,:])
                else:
                    y_true_non_converters.append(self.labels[id])
                    Y_pred_non_converters.append(Y_pred[count, :])
            else:
                y_true_converters.append(self.labels[id])
                Y_pred_converters.append(Y_pred[count, :])
            count += 1

        Y_pred_converters = np.asarray(Y_pred_converters)
        Y_pred_non_converters = np.asarray(Y_pred_non_converters)

        if config.ordered_softmax == True:
            ordinal_to_prob = np.zeros((Y_pred.shape[0], Y_pred.shape[1]+1))
            ordinal_to_prob[:,0] = 1 - np.square(Y_pred[:,0]) - Y_pred[:,1]
            ordinal_to_prob[:,1] = 1 - np.square(1 - Y_pred[:, 0]) - np.square(Y_pred[:,1])
            ordinal_to_prob[:,2] = 1 - (1 - Y_pred[:, 0]) - np.square(1 - Y_pred[:,1])
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


        # calculate acc based on confusion matrix
        conf = confusion_matrix(y_true, y_pred) # NOTE: here in EpochPerformance.py, the order (y_true, y_pred) is different than in main.py (hardEstimClass, true_labels)
        acc = np.trace(conf) / float(np.sum(conf))

        if self.name == "trn" and config.task == "CN-MCI-AD":
            if sum(conf[:,0]) == 0 or sum(conf[:,1]) == 0 or sum(conf[:,2]) == 0:
                config.weird_predictions = True
            else:
                config.weird_predictions = False

        conf_converters = confusion_matrix(y_true_converters, y_pred_converters)  # NOTE: here in EpochPerformance.py, the order (y_true, y_pred) is different than in main.py (hardEstimClass, true_labels)
        acc_converters = np.trace(conf_converters) / float(np.sum(conf_converters))

        conf_non_converters = confusion_matrix(y_true_non_converters, y_pred_non_converters)  # NOTE: here in EpochPerformance.py, the order (y_true, y_pred) is different than in main.py (hardEstimClass, true_labels)
        acc_non_converters = np.trace(conf_non_converters) / float(np.sum(conf_non_converters))

        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(y_true, y_pred)
        try:
            ari_converters = adjusted_rand_score(y_true_converters, y_pred_converters)
        except:
            (print("ARI could not be calculated"))
            ari_converters = 0
        ari_non_converters = adjusted_rand_score(y_true_non_converters, y_pred_non_converters)

        score = self.model.evaluate_generator(generator=self.generator, verbose=1)
        loss = score[0]

        if config.task == "AD" or config.task == "MCI":
            # calculate auc based on prediction estimates
            auc = roc_auc_score(y_true, Y_pred[:,1])

            # calculate sens + spec based on confusion matrix
            sens = conf[1, 1] / (conf[1, 1] + conf[1, 0])
            spec = conf[0, 0] / (conf[0, 0] + conf[0, 1])

        elif config.task == "CN-MCI-AD":

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
                (print("AUC could not be calculated"))
                auc_converters = 0

            # calculate BCA based on prediction estimates
            true_labels = np.asarray(y_true_converters)
            try:
                bca_converters = calcBCA(hardEstimClass, true_labels, nrClasses)
            except:
                (print("BCA could not be calculated"))
                bca_converters = 0

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


        if config.task == "AD" or config.task == "MCI":
            print(f"Epoch {epoch+1:03d}: {self.name}_auc: {auc:.4f} - {self.name}_sens: {sens:.4f} - {self.name}_spec: {spec:.4f} - {self.name}_acc: {acc:.4f}")
            self.auc_list.append(auc)
            self.sens_list.append(sens)
            self.spec_list.append(spec)
            self.acc_list.append(acc)

            # keep track of validation AUC for best model + early stopping
            if self.name is "val":

                # save model if best validation auc
                if auc is max(self.auc_list):
                    filepath = self.results_dir + f"/weights-improvement-e{epoch + 1:02d}-auc{auc:.2f}.hdf5"
                    self.model.save(filepath)
                    self.best_epoch = epoch
                    print(f"Epoch {epoch + 1:03d}: {self.name}_auc improved to {auc:.4f}, saving model to {filepath}\n")
                else:
                    no_improvement = epoch - self.best_epoch
                    print(
                        f"Epoch {epoch + 1:03d}: {self.name}_auc did not improve from {max(self.auc_list):.4f} for {no_improvement:03d} epochs\n")

                    # stop training after # epochs without improvement
                    if config.early_stopping and no_improvement is config.es_patience:
                        print(
                            f"Epoch {epoch + 1:03d}: {no_improvement:03d} epochs without AUC improvement - early stopping\n")
                        self.model.stop_training = True

            return

        elif config.task == "CN-MCI-AD":
            print(f"Epoch {epoch+1:03d}: {self.name}_acc: {acc:.4f} - {self.name}_mauc: {mauc:.4f} - {self.name}_bca: {bca:.4f} - {self.name}_loss: {loss:.4f}")
            self.mauc_list.append(mauc)
            self.bca_list.append(bca)
            self.acc_list.append(acc)
            self.loss_list.append(loss)
            self.auc_list_converters.append(auc_converters)
            self.bca_list_converters.append(bca_converters)
            self.mauc_list_non_converters.append(mauc_non_converters)
            self.bca_list_non_converters.append(bca_non_converters)
            self.acc_list_converters.append(acc_converters)
            self.acc_list_non_converters.append(acc_non_converters)
            #self.acc_list.append(acc)
            self.ari_list.append(ari)
            self.ari_list_converters.append(ari_converters)
            self.ari_list_non_converters.append(ari_non_converters)

            # keep track of validation MAUC for best model + early stopping
            if self.name is "val":
                # save model if best validation mauc or loss
                if config.stopping_based_on == "MAUC":
                    if mauc is max(self.mauc_list):
                        if epoch+1 < 100:
                            filepath = self.results_dir + f"/z-e0{epoch+1:02d}-loss{loss:.2f}-mauc{mauc:.2f}-bca{bca:.2f}-acc_c{acc_converters:.2f}-bca_c{bca_converters:.2f}-mauc_nc{mauc_non_converters:.2f}-bca_nc{bca_non_converters:.2f}.hdf5"
                        else:
                            filepath = self.results_dir + f"/z-e{epoch+1:02d}-loss{loss:.2f}-mauc{mauc:.2f}-bca{bca:.2f}-acc_c{acc_converters:.2f}-bca_c{bca_converters:.2f}-mauc_nc{mauc_non_converters:.2f}-bca_nc{bca_non_converters:.2f}.hdf5"
                        self.model.save(filepath)
                        self.best_epoch = epoch
                        print(f"Epoch {epoch+1:03d}: {self.name}_mauc improved to {mauc:.4f}, saving model to {filepath}\n")
                    else:
                        no_improvement = epoch - self.best_epoch
                        print(f"Epoch {epoch+1:03d}: {self.name}_mauc did not improve from {max(self.mauc_list):.4f} for {no_improvement:03d} epochs\n")

                        # stop training after # epochs without improvement
                        if config.early_stopping and no_improvement is config.es_patience:
                            print(f"Epoch {epoch+1:03d}: {no_improvement:03d} epochs without MAUC improvement - early stopping\n")
                            self.model.stop_training = True
                elif config.stopping_based_on == "Loss":
                    #x = 0
                    if loss is min(self.loss_list): #and x == 1:
                        if epoch+1 < 100:
                            filepath = self.results_dir + f"/z-e0{epoch+1:02d}-loss{loss:.2f}-mauc{mauc:.2f}-bca{bca:.2f}-acc_c{acc_converters:.2f}-bca_c{bca_converters:.2f}-mauc_nc{mauc_non_converters:.2f}-bca_nc{bca_non_converters:.2f}.hdf5"
                        else:
                            filepath = self.results_dir + f"/z-e{epoch+1:02d}-loss{loss:.2f}-mauc{mauc:.2f}-bca{bca:.2f}-acc_c{acc_converters:.2f}-bca_c{bca_converters:.2f}-mauc_nc{mauc_non_converters:.2f}-bca_nc{bca_non_converters:.2f}.hdf5"
                        self.model.save(filepath)
                        self.best_epoch = epoch
                        print(f"Epoch {epoch+1:03d}: {self.name}_loss improved to {loss:.4f}, saving model to {filepath}\n")
                    else:
                        no_improvement = epoch - self.best_epoch
                        print(f"Epoch {epoch+1:03d}: {self.name}_loss did not improve from {min(self.loss_list):.4f} for {no_improvement:03d} epochs\n")

                        # stop training after # epochs without improvement
                        if config.early_stopping and no_improvement is config.es_patience:

                            if self.best_epoch < 1 or config.weird_predictions == True:
                                print(f"Epoch {epoch+1:03d}: {no_improvement:03d} epochs without loss improvement - RESTART TRAINING WITH NEW WEIGHTS\n")
                                config.retrain = True
                            else:
                                print(f"Epoch {epoch+1:03d}: {no_improvement:03d} epochs without loss improvement - early stopping\n")
                                config.retrain = False

                            self.model.stop_training = True

            return

    def on_train_end(self, logs={}):
        return

