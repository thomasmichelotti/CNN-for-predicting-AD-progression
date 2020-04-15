import numpy as np
import pandas as pd
import os
from keras.engine.saving import load_model
import keras
import h5py


def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)

# choose specifications
location = "server"              # local / server
task = "CN-MCI-AD"                              # AD (AD vs. CN) / MCI (pMCI vs. sMCI) / CN-MCI-AD (AD vs. MCI vs. CN)
model = "allCNN"                                # allCNN / allRNN? / CNN-RNN?
roi = "GM_f4"                                   # T1_WB / T1m_WB / GM_WB / GM_f4 (faster for quick runs)
non_image_data = True                                                                                                                   # CHANGE THIS ONE

# choose model directory
if location == "local":
    dir = f"/path/to/D_OutputFiles/B_Output/"
elif location == "server":
    dir = f"/path/to/D_OutputFiles/B_Output/"
model_and_interval = "2020-02-17T09-20_GM_f4_CN-MCI-AD_allCNN_pat20_non_imaging_interval_0_1.25_30Fold/"
new_dir = dir + model_and_interval

# choose PSI data
if location == "local":
    data_dir = f"/path/to/A_DataFiles/PreprocessedDataParelsnoer/newPatients/interval_0_1.25/"
    image_dir = f"/path/to/A_DataFiles/data_PSI_GM_f4/"
elif location == "server":
    data_dir = f"/path/to/A_DataFiles/PreprocessedDataParelsnoer/newPatients/interval_0_1.25/"
    image_dir = f"/path/to/A_DataFiles/data_PSI_GM_f4/"


# create directory for PSI results
if location == "local":
    output_dir = f"/path/to/D_OutputFiles/E_PSI_output/"
elif location == "server":
    output_dir = f"/path/to/D_OutputFiles/E_PSI_output/"
output_dir = output_dir + model_and_interval
create_data_directory(output_dir)

# get dimensions of images
if location == "local":
    with h5py.File(image_dir + "PSI14683.h5py", 'r') as hf:
        x = hf["PSI14683"][:]
elif location == "server":
    with h5py.File(image_dir + "PSI14683.h5py", 'r') as hf:
        x = hf["PSI14683"][:]
input_shape = x.shape
if non_image_data == True:
    feature_list = []
    feature_list.append("Time_feature")
    feature_list.append("Current_diagnosis")
    non_image_input_shape = len(feature_list)
    dim_non_imaging_features = (non_image_input_shape, )