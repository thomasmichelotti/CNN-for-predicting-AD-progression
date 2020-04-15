import config_PSI

import time
import numpy as np

from contextlib import redirect_stdout
from keras.utils import plot_model

def save_results_PSI(results, start, start_localtime, end, end_localtime):
    """
    This function manages the creation of a report in text format of the results of the cross-validation of
    a model. The average loss, acc, AUC, sens and spec are averaged over folds and reported for the evaluation
    of train, validation and test data. This final report is saved as 'results_cross_val.txt' in the output dir.
    """

    final_results = "FINAL RESULTS\n\n"\
                    "Task: 		    " + config_PSI.task + "\n" \
                    "Model: 		" + config_PSI.model + "\n" \
                    "ROI: 		    " + config_PSI.roi + "\n" \
                    "Location: 	    " + config_PSI.location + "\n" \
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


    # write to file and save
    file = open(config_PSI.output_dir + 'final_results.txt', 'w')
    file.write(final_results), print(final_results)
    file.close()