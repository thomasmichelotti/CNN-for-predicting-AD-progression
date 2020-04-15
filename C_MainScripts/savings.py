# import configuration file
import config
#import config.retry_ADNI

import time
import numpy as np

from contextlib import redirect_stdout
from keras.utils import plot_model


def save_results(results, start, start_localtime, end, end_localtime):
    """
    This function manages the creation of a report in text format of the results of the cross-validation of
    a model. The average loss, acc, AUC, sens and spec are averaged over folds and reported for the evaluation
    of train, validation and test data. This final report is saved as 'results_cross_val.txt' in the output dir.
    """

    if config.task == "CN-MCI-AD":
        # save final results
        if config.leaderboard:
            final_results = "FINAL RESULTS\n\n"\
                            "Task: 		    " + config.task + "\n" \
                            "Model: 		" + config.model + "\n" \
                            "ROI: 		    " + config.roi + "\n" \
                            "Location: 	    " + config.location + "\n" \
                            "Comments: 	    " + config.comments + "\n" \
                            "Leaderboard:   " + str(config.leaderboard) + "\n\n" \
                            "TRAIN\n" \
                            "    loss: 	                " + str(np.mean(results["train"]["loss"])) + "\n" \
                            "    MAUC: 	                " + str(np.mean(results["train"]["mauc"])) + ' (' + str(np.std(results['train']['mauc'])) + ')'  + "\n" \
                            "    BCA: 	                " + str(np.mean(results["train"]["bca"])) + ' (' + str(np.std(results['train']['bca'])) + ')'  + "\n" \
                            "    ACC: 	                " + str(np.mean(results["train"]["acc"])) + ' (' + str(np.std(results['train']['acc'])) + ')' + "\n" \
                            "    ARI: 	                " + str(np.mean(results["train"]["ari"])) + ' (' + str(np.std(results['train']['ari'])) + ')' + "\n" \
                            "    AUC CONVERTERS: 	    " + str(np.mean(results["train"]["auc_converters"])) + ' (' + str(np.std(results['train']['auc_converters'])) + ')'  + "\n" \
                            "    BCA CONVERTERS: 	    " + str(np.mean(results["train"]["bca_converters"])) + ' (' + str(np.std(results['train']['bca_converters'])) + ')'  + "\n" \
                            "    ACC CONVERTERS: 	    " + str(np.mean(results["train"]["acc_converters"])) + ' (' + str(np.std(results['train']['acc_converters'])) + ')' + "\n" \
                            "    ARI CONVERTERS: 	    " + str(np.mean(results["train"]["ari_converters"])) + ' (' + str(np.std(results['train']['ari_converters'])) + ')' + "\n" \
                            "    MAUC NON CONVERTERS: 	" + str(np.mean(results["train"]["mauc_non_converters"])) + ' (' + str(np.std(results['train']['mauc_non_converters'])) + ')'  + "\n" \
                            "    BCA NON CONVERTERS: 	" + str(np.mean(results["train"]["bca_non_converters"])) + ' (' + str(np.std(results['train']['bca_non_converters'])) + ')'  + "\n" \
                            "    ACC NON CONVERTERS: 	" + str(np.mean(results["train"]["acc_non_converters"])) + ' (' + str(np.std(results['train']['acc_non_converters'])) + ')' + "\n" \
                            "    ARI NON CONVERTERS: 	" + str(np.mean(results["train"]["ari_non_converters"])) + ' (' + str(np.std(results['train']['ari_non_converters'])) + ')' + "\n" \
                            "VALIDATION\n" \
                            "    loss: 	                " + str(np.mean(results["validation"]["loss"])) + "\n" \
                            "    MAUC: 	                " + str(np.mean(results["validation"]["mauc"])) + ' (' + str(np.std(results['validation']['mauc'])) + ')'  + "\n" \
                            "    BCA: 	                " + str(np.mean(results["validation"]["bca"])) + ' (' + str(np.std(results['validation']['bca'])) + ')'  + "\n" \
                            "    ACC: 	                " + str(np.mean(results["validation"]["acc"])) + ' (' + str(np.std(results['validation']['acc'])) + ')' + "\n" \
                            "    ARI: 	                " + str(np.mean(results["validation"]["ari"])) + ' (' + str(np.std(results['validation']['ari'])) + ')' + "\n" \
                            "    AUC CONVERTERS: 	    " + str(np.mean(results["validation"]["auc_converters"])) + ' (' + str(np.std(results['validation']['auc_converters'])) + ')'  + "\n" \
                            "    BCA CONVERTERS: 	    " + str(np.mean(results["validation"]["bca_converters"])) + ' (' + str(np.std(results['validation']['bca_converters'])) + ')'  + "\n" \
                            "    ACC CONVERTERS: 	    " + str(np.mean(results["validation"]["acc_converters"])) + ' (' + str(np.std(results['validation']['acc_converters'])) + ')' + "\n" \
                            "    ARI CONVERTERS: 	    " + str(np.mean(results["validation"]["ari_converters"])) + ' (' + str(np.std(results['validation']['ari_converters'])) + ')' + "\n" \
                            "    MAUC NON CONVERTERS: 	" + str(np.mean(results["validation"]["mauc_non_converters"])) + ' (' + str(np.std(results['validation']['mauc_non_converters'])) + ')'  + "\n" \
                            "    BCA NON CONVERTERS: 	" + str(np.mean(results["validation"]["bca_non_converters"])) + ' (' + str(np.std(results['validation']['bca_non_converters'])) + ')'  + "\n" \
                            "    ACC NON CONVERTERS: 	" + str(np.mean(results["validation"]["acc_non_converters"])) + ' (' + str(np.std(results['validation']['acc_non_converters'])) + ')' + "\n" \
                            "    ARI NON CONVERTERS: 	" + str(np.mean(results["validation"]["ari_non_converters"])) + ' (' + str(np.std(results['validation']['ari_non_converters'])) + ')' + "\n" \
                            "TEST\n" \
                            "    MAUC: 	                " + str(np.mean(results["test"]["mauc"])) + ' (' + str(np.std(results['test']['mauc'])) + ')'  + "\n" \
                            "    BCA: 	                " + str(np.mean(results["test"]["bca"])) + ' (' + str(np.std(results['test']['bca'])) + ')'  + "\n" \
                            "    ACC: 	                " + str(np.mean(results["test"]["acc"])) + ' (' + str(np.std(results['test']['acc'])) + ')' + "\n" \
                            "    ARI: 	                " + str(np.mean(results["test"]["ari"])) + ' (' + str(np.std(results['test']['ari'])) + ')' + "\n" \
                            "    AUC CONVERTERS: 	    " + str(np.mean(results["test"]["auc_converters"])) + ' (' + str(np.std(results['test']['auc_converters'])) + ')'  + "\n" \
                            "    BCA CONVERTERS: 	    " + str(np.mean(results["test"]["bca_converters"])) + ' (' + str(np.std(results['test']['bca_converters'])) + ')'  + "\n" \
                            "    ACC CONVERTERS: 	    " + str(np.mean(results["test"]["acc_converters"])) + ' (' + str(np.std(results['test']['acc_converters'])) + ')' + "\n" \
                            "    ARI CONVERTERS: 	    " + str(np.mean(results["test"]["ari_converters"])) + ' (' + str(np.std(results['test']['ari_converters'])) + ')' + "\n" \
                            "    MAUC NON CONVERTERS: 	" + str(np.mean(results["test"]["mauc_non_converters"])) + ' (' + str(np.std(results['test']['mauc_non_converters'])) + ')'  + "\n" \
                            "    BCA NON CONVERTERS: 	" + str(np.mean(results["test"]["bca_non_converters"])) + ' (' + str(np.std(results['test']['bca_non_converters'])) + ')'  + "\n" \
                            "    ACC NON CONVERTERS: 	" + str(np.mean(results["test"]["acc_non_converters"])) + ' (' + str(np.std(results['test']['acc_non_converters'])) + ')' + "\n" \
                            "    ARI NON CONVERTERS: 	" + str(np.mean(results["test"]["ari_non_converters"])) + ' (' + str(np.std(results['test']['ari_non_converters'])) + ')' + "\n" \
                            "Start: 		" + str(time.strftime("%m/%d/%Y, %H:%M:%S", start_localtime)) + "\n" \
                            "End: 		    " + str(time.strftime("%m/%d/%Y, %H:%M:%S", end_localtime)) + "\n" \
                            "Run time: 	    " + str(round((end - start), 2)) + " seconds"
        else:
            final_results = "FINAL RESULTS\n\n"\
                            "Task: 		" + config.task + "\n" \
                            "Model: 		" + config.model + "\n" \
                            "ROI: 		" + config.roi + "\n" \
                            "Location: 	" + config.location + "\n" \
                            "Comments: 	" + config.comments + "\n" \
                            "Leaderboard:  " + str(config.leaderboard) + "\n\n" \
                            "TRAIN\n" \
                            "    loss: 	" + str(results["train"]["loss"]) + "\n" \
                            "    MAUC: 	" + str(results["train"]["mauc"]) + "\n" \
                            "    BCA: 	" + str(results["train"]["bca"]) + "\n" \
                            "VALIDATION\n" \
                            "    loss: 	" + str(results["validation"]["loss"]) + "\n" \
                            "    MAUC: 	" + str(results["validation"]["mauc"]) + "\n" \
                            "    BCA: 	" + str(results["validation"]["bca"]) + "\n" \
                            "Start: 		" + str(time.strftime("%m/%d/%Y, %H:%M:%S", start_localtime)) + "\n" \
                            "End: 		" + str(time.strftime("%m/%d/%Y, %H:%M:%S", end_localtime)) + "\n" \
                            "Run time: 	" + str(round((end - start), 2)) + " seconds"

        # write to file and save
        file = open(config.output_dir + 'final_results.txt', 'w')
        file.write(final_results), print(final_results)
        file.close()
    else:
        # save final results
        final_results = "RESULTS " + str(config.k_cross_validation) + "-FOLD CROSS VALIDATION\n\n" \
                        "Task: 		" + config.task + "\n" \
                        "Model: 		" + config.model + "\n" \
                        "ROI: 		" + config.roi + "\n" \
                        "Location: 	" + config.location + "\n" \
                        "Comments: 	" + config.comments + "\n\n" \
                        "TRAIN\n" \
                        "    loss: 	" + str(np.mean(results["train"]["loss"])) + "\n" \
                        "    acc: 	" + str(np.mean(results["train"]["acc"])) + "\n" \
                        "    AUC: 	" + str(np.mean(results["train"]["auc"])) + "\n" \
                        "    sens: 	" + str(np.mean(results["train"]["sensitivity"])) + "\n" \
                        "    spec:	" + str(np.mean(results["train"]["specificity"])) + "\n\n"\
                        "VALIDATION\n" \
                        "    loss: 	" + str(np.mean(results["validation"]["loss"])) + "\n" \
                        "    acc: 	" + str(np.mean(results["validation"]["acc"])) + "\n" \
                        "    AUC: 	" + str(np.mean(results["validation"]["auc"])) + "\n" \
                        "    sens: 	" + str(np.mean(results["validation"]["sensitivity"])) + "\n" \
                        "    spec: 	" + str(np.mean(results["validation"]["specificity"])) + "\n\n"\
                        "TEST\n" \
                        "    loss: 	" + str(np.mean(results["test"]["loss"])) + "\n" \
                        "    acc: 	" + str(np.mean(results["test"]["acc"])) + "\n" \
                        "    AUC: 	" + str(np.mean(results["test"]["auc"])) + "\n" \
                        "    sens: 	" + str(np.mean(results["test"]["sensitivity"])) + "\n" \
                        "    spec: 	" + str(np.mean(results["test"]["specificity"])) + "\n\n" \
                        "Start: 		" + str(time.strftime("%m/%d/%Y, %H:%M:%S", start_localtime)) + "\n" \
                        "End: 		" + str(time.strftime("%m/%d/%Y, %H:%M:%S", end_localtime)) + "\n" \
                        "Run time: 	" + str(round((end - start), 2)) + " seconds"

        # write to file and save
        file = open(config.output_dir + 'results_cross_val.txt', 'w')
        file.write(final_results), print(final_results)
        file.close()


def save_DL_model(model):
    """
    Saves a picture and summary of the used model in the output dir.
    """

    plot_model(model, show_shapes=True, to_file=config.output_dir + 'model.png')
    with open(config.output_dir + 'modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()



