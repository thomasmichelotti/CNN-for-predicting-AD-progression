"""
CONFIGURATION FILE

This file contains all deep learning settings for the TADPOLE prediction of AD clinical diagnosis progression
"""

# Preprocessing
#preprocessing = 0                               # 0 if not necessary, 1 if necessary (only once)

import C_MainScripts.ordinal_categorical_crossentropy as OCC
import keras

# General settings
leaderboard = 1                                 # 0 (Submit for TADPOLE D4) / 1 (Test on Leaderboard data)
d3 = 0                                          # 0 (D2 -> longitudinal) / 1 (D3 -> cross-sectional)
task = "CN-MCI-AD"                              # AD (AD vs. CN) / MCI (pMCI vs. sMCI) / CN-MCI-AD (AD vs. MCI vs. CN)
model = "allCNN"                                # allCNN / allRNN? / CNN-RNN?
roi = "GM_f4"                                   # T1_WB / T1m_WB / GM_WB / GM_f4 (faster for quick runs)

non_image_data = True                          # True / False (only for CN-MCI-AD)
only_non_imaging = False
include_time_interval = True
include_mmse = False
include_adas13 = False
include_current_diagnosis = True
if non_image_data == True:
    non_imaging_filename = "all_non_image_features.csv"
    non_image_input_shape = 0
    feature_list = []
    if include_time_interval == True:
        non_image_input_shape += 1
        feature_list.append("Time_feature")
    if include_mmse == True:
        non_image_input_shape += 1
        feature_list.append("MMSE")
    if include_adas13 == True:
        non_image_input_shape += 1
        feature_list.append("ADAS13")
    if include_current_diagnosis == True:
        non_image_input_shape += 1
        feature_list.append("Current_diagnosis")

location = "local"                              # local / server
reproducible = False                             # True (set global seeds to get reproducible results, which is substantially slower because no parallel computing can be used) / False
stopping_based_on = "Loss"                      # Loss or MAUC (on the validation set), only works for CN-MCI-AD

partially_trained_weights = False
partially_trained_weights_dir = f"path/to/D_OutputFiles/B_Output/xyz.hdf5"
freeze_weights = True
fully_trained = False
fully_trained_model_dir = f"path/to/D_OutputFiles/B_Output/xyz.hdf5"

lr_optimiser = "Adam"                           # Adam / Adadelta
loss_function = OCC.customLossFunction          # keras.losses.categorical_crossentropy, OCC.customLossFunction, keras.losses.mean_squared_error, ...
interval_CNN = True
if interval_CNN == True:                        # lb-ub intervals available: 0-1.25, 1.25-2.25, 2.25-3.25, 3.25-5.25
    lb = 0
    ub = 1.25
    interval = f"interval_{lb}_{ub}"
test_deeper_cnn = False
test_broader_cnn = False
TADPOLE_setup = False
ordered_softmax = False


extra_penalty_misclassified_converters = True
differentiate_mci_ad_penalty = True
converter_penalty = 0
converter_penalty_mci = 0
converter_penalty_ad = 0
ordinal_penalty = True

dynamic_batch_size = True

# pre-training
pre_train = False                               # if True: uses pre-trained model
pre_train_model = f"path/to/D_OutputFiles/B_Output/xyz.hdf5"   # path to pre-trained model
freeze = False                                  # if True: freezes layers of the network
freeze_until = 7                                # Freeze until this layer

# MCI evaluation (use with MCI_crossval.py)
test_only = False                                # only evaluation, no training
pretrain_path = "/path/to/crossvalidationfolder"# path to pre-trained AD model
all_data = False                                 # use all data in once for normalisation
mean_file = "/path/to/mean/mean.npy"            # path to mean of AD training data for normalisation
std_file = "/path/to/std/std.npy"               # path to std of AD training data for normalisation

# parameters
k_cross_validation = 30
epochs = 500
batch_size = 4
test_size = 0.1                                 # .. % test set
val_size = 0.1                                  # .. % validation set

# split
shuffle_split = True                            # True: (stratified) shuffle split, for MCI testing and quick experiments
                                                # False: (stratified) K fold, for AD training/testing

# augmentation
augmentation = False                            # True: apply data augmentation
class_total = 1000                               # augment to ... images per class
aug_factor = 0.2                                # mix images with factor ... (only used for task AD and MCI)
dirichlet_alpha = 0.1                          # alpha parameter for dirichlet distribution (only used for task CN-MCI-AD)

# params
lr = 0.001                                      # learning rate
rho = 0.7                                       # Standard value for Adam optimizer (?)
epsilon = 1e-8                                  # Standard value for Adam optimizer
decay = 0.0
dropout = 0.30

# callback options
epoch_performance = True                        # True: compute performance measures after every epoch
early_stopping = True                           # True: stops when es_patience is reached without improvement
es_patience = 20
retrain = True                                  # Always keep on True, will be changed later in the scripts
retrain_patience = 6                            # Should be at least 1, a patience of 2 means that there will be a max of 1 reinitialisation of the random weight initialisation
weird_predictions = False                       # Always keep on False, will be changed later in the scripts
lr_scheduler = False                             # True: reduces lr with lr_drop after lr_epochs_drop epochs
lr_drop = 0.5
lr_epochs_drop = 20
tensorboard = False
acc_checkpoint = False                          # Old function for epoch_performance
acc_early_stopping = False

# seed
#if reproducible == True:
fixed_seed = 1                                  # set fixed seed to compare runs, initialisation Keras/Tensorflow, not guaranteed to work on GPU cluster


# regularization
from keras import regularizers                  # Experiment done on regularisation, mostly set to None.
weight_regularize = None                        # regularizers.l2(0.01) / None

# Specify comments
#comments = f"_drop{dropout}_pat{es_patience}_stop{stopping_based_on}_optim{lr_optimiser}"      # additional comments
comments = f"_pat{es_patience}"      # additional comments
if non_image_data == True:
    if only_non_imaging == True:
        comments = comments + f"_only_non_imaging"
    else:
        comments = comments + f"_non_imaging"
if partially_trained_weights == True:
    comments = comments + f"_pretrain"
if interval_CNN == True:
    comments = comments + f"_{interval}"
if test_deeper_cnn == True:
    comments = comments + f"_deeperCNN"
if test_broader_cnn == True:
    comments = comments + f"_broaderCNN"
# if TADPOLE_setup == False:
#     comments = comments + f"_newPatients"
# else:
#     comments = comments + f"_TADPOLEsetup"
if augmentation == True:
    # comments = comments + f"_aug{class_total}"
    comments = comments + f"_aug"
comments = comments + f"_{k_cross_validation}Fold"
if ordered_softmax == True:
    comments = comments + f"_ordinal"

#######################################################################################

# directories

if test_only:
    comments = comments + "_notrain"

import datetime
stamp = datetime.datetime.now().isoformat()[:16]
stamp = stamp.replace(":", "-")


if location == "local":
    # Define input directory
    import os
    str_exp = os.path.dirname(os.path.realpath(__file__))
    os.chdir(str_exp)

    # Define paths to input data directories and files (labels, images, non-imaging data)
    preprocessed_data_dir = f"path/to/A_DataFiles/PreprocessedData/"
    data_dir = f"path/to/A_DataFiles/data_ADNI_{roi}/"
    #subjectfile = "C:/Users/050522/PycharmProjects/CNN_Jara/LOCAL_INPUT_OUTPUT/labels/AllSubjectsDiagnosis.csv"

    # Define paths to output files
    output_dir = f"path/to/D_OutputFiles/B_Output/{stamp}_{roi}_{task}_{model}{comments}/"
    aug_dir = f"path/to/D_OutputFiles/B_Output/{stamp}_{roi}_{task}_{model}{comments}/augmentation/"
    final_output_dir = f"path/to/D_OutputFiles/D_Final_TADPOLE_Submissions/{stamp}_{roi}_{task}_{model}{comments}/"
    str_out_final = f"path/to/D_OutputFiles/D_Final_TADPOLE_Submissions/{stamp}_{roi}_{task}_{model}{comments}/TADPOLE_Submission_EMC-TM.csv"

    # Define other paths
    config_file = f"path/to/config.py"

    if leaderboard:
        final_output_dir = final_output_dir.replace('D_Final_TADPOLE_Submissions', 'C_Leaderboard_Submissions')
        str_out_final = str_out_final.replace('D_Final_TADPOLE_Submissions', 'C_Leaderboard_Submissions')
        str_out_final = str_out_final.replace('_Submission_', '_Submission_Leaderboard_')
    if d3:
        str_out_final = str_out_final.replace('EMC', 'EMC-D3')
    if leaderboard and d3:
        str_out_final = str_out_final.replace('D_Final_TADPOLE_Submissions', 'C_Leaderboard_Submissions')
        str_out_final = str_out_final.replace('_Submission_', '_Submission_Leaderboard_')
        str_out_final = str_out_final.replace('EMC', 'EMC-D3')

else:
    if task == "CN-MCI-AD":
        preprocessed_data_dir = f"path/to/A_DataFiles/PreprocessedData/"
        data_dir = f"path/to/A_DataFiles/data_ADNI_{roi}/"

        # Define paths to output files
        output_dir = f"path/to/D_OutputFiles/B_Output/{stamp}_{roi}_{task}_{model}{comments}/"
        aug_dir = f"{data_dir}/augmentation/"
        final_output_dir = f"path/to/D_OutputFiles/D_Final_TADPOLE_Submissions/{stamp}_{roi}_{task}_{model}{comments}/"
        str_out_final = f"path/to/D_OutputFiles/D_Final_TADPOLE_Submissions/{stamp}_{roi}_{task}_{model}{comments}/TADPOLE_Submission_XYZ.csv"

        # Define other paths
        config_file = f"path/to/config.py"

        if leaderboard:
            final_output_dir = final_output_dir.replace('D_Final_TADPOLE_Submissions', 'C_Leaderboard_Submissions')
            str_out_final = str_out_final.replace('D_Final_TADPOLE_Submissions', 'C_Leaderboard_Submissions')
            str_out_final = str_out_final.replace('_Submission_', '_Submission_Leaderboard_')
        if d3:
            str_out_final = str_out_final.replace('EMC', 'EMC-D3')
        if leaderboard and d3:
            str_out_final = str_out_final.replace('D_Final_TADPOLE_Submissions', 'C_Leaderboard_Submissions')
            str_out_final = str_out_final.replace('_Submission_', '_Submission_Leaderboard_')
            str_out_final = str_out_final.replace('EMC', 'EMC-D3')

    else:
        all_results_dir = f"path/to/results/"
        output_dir = f"{all_results_dir}{stamp}_{roi}_{task}_{model}{comments}/"
        #data_dir = f"path/to/data_{roi}/"
        #aug_dir = f"{data_dir}/augmentation/"
        #config_file = f"path/to/cnn-for-ad-classification/config.py"
        #subjectfile = f"path/to/labels/AllSubjectsDiagnosis.csv"
        #subjectfile = f"path/to/labels_new/labels_adni.csv"

# set classes
if task == "AD":
    class0 = "CN"
    class1 = "AD"
elif task == "MCI":
    class0 = "MCI-s"
    class1 = "MCI-c"
else:
    class0 = "CN"
    class1 = "MCI"
    class2 = "AD"

# set input shape
import numpy as np
import h5py

if task != "AD" and task != "MCI":
    with h5py.File(data_dir + "002_S_0295_bl.h5py", 'r') as hf:
        x = hf["002_S_0295_bl"][:]
    input_shape = x.shape
else:
    input_shape = np.load(data_dir + "002_S_0295.npy").shape
fpr_interval = np.linspace(0, 1, 100)           # Range of false positive rates used for generating ROC curve.

