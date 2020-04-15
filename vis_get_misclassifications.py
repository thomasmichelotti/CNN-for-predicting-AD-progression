import csv
import os
import sys
from random import shuffle
import math
import numpy as np
import pandas as pd
from keras.engine.saving import load_model
import h5py
import os

# import configuration parameters
from vis_config import roi, task, label, data_set, val, server, class_limit, interval, model_id, repetition_id, non_image_features


def main():
    """
    This script should be run before 'vis_main.py' to get a list of correctly and miss classified subjects for
    a specific class. These lists are saved in the info path for compiling the Grad-CAM images and are used
    for the class-averaged computation of these images.

    Because not more than 10 images can be compiled at once the lists with subjects are build up as multiple
    seperate lists: e.g. corr_subs[i] in which i is the amount of seperate subject splits. In this way the
    'vis_main.py' script can be run several times using every split with subjects and can finally average the
    Grad-CAMs of these splits with 'vis_average.py'.

    A Grad-CAM info directory should be created in which the model, mean, std and train-test files are stored.
    These will be accessed by the current visualization scripts. This script will store several subject files
    in this info folder including all the correct and miss classified subjects per class, which can be accessed
    by the other visualization scripts.

    """

    # set paths (when running on server based on data dir + job nr input)
    if server:
        data_path = sys.argv[1] + "/"
        info_path = f"/path/to/D_OutputFiles/B_Output/{model_id}/"
        subjectfile = f"/path/to/A_DataFiles/PreprocessedData/newPatients/{interval}/labels.csv"
        features = f"/path/to/A_DataFiles/PreprocessedData/newPatients/{interval}/all_non_image_features.csv"
        output_path = f"/path/to/D_OutputFiles/F_GradCAM_output/A_Misclassifications/{model_id}_k{repetition_id}/"
    else:
        data_path = f"/path/to/A_DataFiles/data_ADNI_GM_f4/"
        info_path = f"/path/to/D_OutputFiles/B_Output/{model_id}/"
        subjectfile = f"/path/to/A_DataFiles/PreprocessedData/newPatients/{interval}/labels.csv"
        features = f"/path/to/A_DataFiles/PreprocessedData/newPatients/{interval}/all_non_image_features.csv"
        output_path = f"/path/to/D_OutputFiles/F_GradCAM_output/A_Misclassifications/{model_id}_k{repetition_id}/"

    # set model information files
    mean_file = info_path + "k" + repetition_id + "/mean.npy"
    std_file = info_path + "k" + repetition_id + "/std.npy"
    L = os.listdir(info_path + "k" + repetition_id + "/")
    L.sort()
    model_file = info_path + "k" + repetition_id + "/" + L[-1]
    set_file = f"{info_path}train_val.npy" if val or data_set == "train" else f"{info_path}train_test.npy"

    if non_image_features == True:
        all_features = pd.read_csv(features)
        non_image_mean_file = info_path + "k" + repetition_id + "/non_image_mean.npy"
        non_image_std_file = info_path + "k" + repetition_id + "/non_image_std.npy"
        non_image_mean = np.load(non_image_mean_file)
        non_image_std = np.load(non_image_std_file)
    else:
        all_features = 0
        non_image_mean = 0
        non_image_std = 0

    # set classes
    if task == "AD":
        class0 = "CN"
        class1 = "AD"
        class2 = "XXX"
    elif task == "MCI":
        class0 = "MCI-s"
        class1 = "MCI-c"
        class2 = "XXX"
    else:
        class0 = "CN"
        class1 = "MCI"
        class2 = "AD"

    # split subjects based on classes: {"AD": [1, 2, ...], "CN" : [3, 4, ...]}
    classes_split = split_classes(subjectfile, set_file, class0, class1, class2, repetition_id)

    # get mean + std + model
    mean = np.load(mean_file)
    std = np.load(std_file)
    model = load_model(model_file, compile=False)

    # get model predictions
    corr_class, miss_class = count_missclass(classes_split[label], data_path, mean, model, std, class0, class1, class2, all_features, non_image_mean, non_image_std)

    # split subjects based on correct or miss classification
    corr_subs, k_splits_corr = split_subjects(corr_class)
    miss_subs, k_splits_miss = split_subjects(miss_class)

    file = open(f"{output_path}k_splits_{label}_{data_set}.txt", 'w')
    file.write(f"Number of splits for correctly classified subjects: {k_splits_corr}")
    file.write(f"Number of splits for misclassified subjects: {k_splits_miss}")
    file.close()

    print(f"\nCORRECT: {len(corr_subs)} splits of {len(corr_subs[0])} subjects")
    print(f"\nMISS: {len(miss_subs)} splits of {len(miss_subs[0])} subjects")

    # save the correct and miss classified subjects of a model for a specific class
    create_data_directory(output_path)
    np.save(f"{output_path}correct_classified_subjects_{label}_{data_set}.npy", corr_subs)
    np.save(f"{output_path}miss_classified_subjects_{label}_{data_set}.npy", miss_subs)

    print('\nend')


def split_classes(subjectfile, set_file, class0, class1, class2, repetition_id):
    """
    Splits subjects of the test group in 2 or 3 classes
    Returns a dictionary: {"AD": [sub1, sub2, ...], "CN": [sub5, sub6, ...]}
    """
    # get test subjects
    subs = np.load(set_file, allow_pickle=True).item()[data_set][int(repetition_id)]
    print(f"\nIn total {len(subs)} subjects present in {data_set} set")

    # split subjects based on class
    if task == "CN-MCI-AD":
        classes_split = {class0: [], class1: [], class2: []}
    else:
        classes_split = {class0: [], class1: []}
    file = open(subjectfile, 'r')
    for line in csv.reader(file):
        if line[3] == '1.0' and np.isin(line[2], subs):
            classes_split[class0].append(line[2])
        elif line[3] == '2.0' and np.isin(line[2], subs):
            classes_split[class1].append(line[2])
        elif line[3] == '3.0' and np.isin(line[2], subs):
            classes_split[class2].append(line[2])

    print("\nCATEGORIES")
    for cat in classes_split:
        print(f"    {cat} : {len(classes_split[cat])}")

    shuffle(classes_split[class0])
    shuffle(classes_split[class1])
    if task == "CN-MCI-AD":
        shuffle(classes_split[class2])

    file.close()

    return classes_split


def count_missclass(subjects, data_path, mean, model, std, class0, class1, class2, all_features, non_image_mean, non_image_std):
    """
    Get the model predictions for multiple input images. Splits the correctly and miss classified
    subjects based on these predictions.
    """

    corr_class = []
    miss_class = []

    for i, subject in enumerate(subjects):
        this_row = pd.DataFrame(all_features[all_features["Link"] == subject])
        idx = this_row.index.item()
        ptid_viscode = this_row.loc[idx, "PTID_VISCODE"]

        img_file = f"{data_path}{ptid_viscode}.h5py"
        print(f"\n{i} - Working on subject: {ptid_viscode} - with true label: {label}")

        # load + standardize image
        image = load_image(img_file, mean, std, ptid_viscode)

        exp_im = np.expand_dims(image, axis=0)
        exp_im = np.expand_dims(exp_im, axis=5)

        if non_image_features == True:
            X_non_image = np.empty((1, 2,))

            feature_list = []
            feature_list.append("Time_feature")
            feature_list.append("Current_diagnosis")
            non_image_columns = this_row[this_row.columns.intersection(feature_list)].to_numpy()
            X_non_image[0,] = non_image_columns[0]

            X_non_image[0,] = np.subtract(X_non_image[0,], non_image_mean)
            X_non_image[0,] = np.divide(X_non_image[0,], non_image_std)

            X = [np.array(exp_im), np.array(X_non_image)]
            predictions = model.predict(X)
        else:
            predictions = model.predict(exp_im)

        cls = np.argmax(predictions)
        if cls == 0:
            class_name = class0
        elif cls == 1:
            class_name = class1
        elif cls == 2:
            class_name = class2
        print(f'\tModel prediction:\n\t\t{class_name}\twith probability {predictions[0][cls]:.4f}')

        # only process if correct classification
        if class_name == label:
            corr_class.append(subject)          # or ptid_viscode?
        else:
            print("Miss classification")
            miss_class.append(subject)

    update = f"\n{label} - correct classifications: {len(corr_class)} - miss classifications: {len(miss_class)}\n\n"
    print(update)

    return corr_class, miss_class


def split_subjects(subjects):
    """
    Splits the list of subjects in k splits.
    Needed when running on the server with WB data, which memory allows max 10 subs at once.
    By splitting the subjects the script can run multiple times, at the end the average is calculated.
    """
    subs_split = []

    if class_limit > len(subjects):
        k_splits = 1
    else:
        k_splits = math.ceil(len(subjects) / class_limit)

    # split subjects in k lists of 10
    for k in range(1, k_splits + 1):
        subs_split.append(subjects[k*class_limit-class_limit:k*class_limit])

    return subs_split, k_splits


def load_image(img_file, mean, std, subject):
    """Load and normalize image"""

    with h5py.File(img_file, 'r') as hf:
        x = hf[subject][:]
    x = np.subtract(x, mean)
    x = np.divide(x, (std + 1e-10))

    return x


def create_data_directory(path):
    """
    Create data path when not existed yet.
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()
