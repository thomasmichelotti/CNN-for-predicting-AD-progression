"""
CONFIGURATION FILE

This file contains all settings for the Grad-CAM computations of the all-CNN trained for MRI AD classification.

"""

roi = "GM_f4"                       # GM_WB / T1_WB / T1m_WB
task = "CN-MCI-AD"                  # AD / MCI / CN-MCI-AD
label = "AD"                        # AD / CN / MCI / MCI-s / MCI-c                                                                 # change
interval = "interval_0_1.25"        # interval_0_1.25 / interval_1.25_2.25 / interval_2.25_3.25                                     # change
model_id = "2020-01-24T17-23_GM_f4_CN-MCI-AD_allCNN_pat20_non_imaging_interval_0_1.25_30Fold"                                       # change
repetition_id = "17"                # 0-1: 17, 1-2: 6, 2-3: 12                                                                      # change
non_image_features = True           # True / False

classification_type = "correct"     # miss / correct
data_set = "test"                   # train / test                                                                                  # change
mask_factor = 0.05                   # only the most important (mask_factor)% of the features are visualised
learning_phase = 0
val = False                         # True: if test set for pre-trained model is needed

gc_layer = 3                     # layer to visualize gradcam
class_limit = 10                    # amount of subjects to be processed per run
run = 0                             # starts at zero

server = True                       # True: if run on server

