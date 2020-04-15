# import configuration file
import config

# set random seed
#from numpy.random import seed
#from tensorflow import set_random_seed
#seed(config.fixed_seed)
#set_random_seed(config.fixed_seed)

import csv
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, ShuffleSplit, KFold
import copy


def create_labels():
    """
    Reading & formatting of the subject IDs and labels

        INPUT:
            subjectfile - should be a .csv file with subject IDs in col 0 and labels in col 1

        OUTPUT:
            partition_labels - format {"CN" : [001, 002, 003], "AD" : [004, 005 ...] }
            labels - format {"001" : 0 , "002" : 1, "003" : 0 ...}
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    # create dicts
    if config.task == "AD" or config.task == "MCI":
        partition_labels = {config.class0: [], config.class1: []}
    else:
        partition_labels = {config.class0: [], config.class1: [], config.class2: []}
    labels = dict()

    # open csv file and write 0 for class0 (CN / MCI-s) and 1 for class1 (AD / MCI-c)
    file = open(config.subjectfile, 'r')
    for line in csv.reader(file):
        if config.task == "CN-MCI-AD":
            if line[1] == "MCI-c" or line[1] == "MCI-s":
                line[1] = "MCI"
            if line[1] == config.class2:
                partition_labels[line[1]].append(line[0])
                labels[line[0]] = 2
        if line[1] == config.class0:
            partition_labels[line[1]].append(line[0])
            labels[line[0]] = 0
        elif line[1] == config.class1:
            partition_labels[line[1]].append(line[0])
            labels[line[0]] = 1

    file.close()

    print("\nCATEGORIES\n")
    for cat in partition_labels:
        print("    " + cat + " : " + str(len(partition_labels[cat])))

    return partition_labels, labels

def create_longitudinal_labels():
    """
    Reading & formatting of the subject IDs and labels

        INPUT:
            subjectfile - should be a .csv file with subject IDs in col 0 and labels in col 1

        OUTPUT:
            partition_labels - format {"CN" : [001, 002, 003], "AD" : [004, 005 ...] }
            labels - format {"001" : 0 , "002" : 1, "003" : 0 ...}
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    # create dicts
    partition_labels = {config.class0: [], config.class1: [], config.class2: []}
    labels = dict()

    # open csv file and write 0 for class0 (CN / MCI-s) and 1 for class1 (AD / MCI-c)
    file = open(config.all_diagnosis, 'r')
    for line in csv.reader(file):
        if line[3] == "1.0":
            partition_labels[config.class0].append(line[1])
            labels[line[1]] = 0
        elif line[3] == "2.0":
            partition_labels[config.class1].append(line[1])
            labels[line[1]] = 1
        elif line[3] == "3.0":
            partition_labels[config.class2].append(line[1])
            labels[line[1]] = 2

    file.close()

    print("\nCATEGORIES\n")
    for cat in partition_labels:
        print("    " + cat + " : " + str(len(partition_labels[cat])))

    return partition_labels, labels


def split_train_test(partition_labels, labels):
    """
    Splits a dataset in a training and test set, based on stratified K fold

        INPUT:
            partition_labels, labels - see 'create_labels()'

        OUTPUT:
            partition_train_test[k] - subjects split in train and test group
            format {"train" : [001, 002, 003], "test" : [004, 005, ...] }
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    partition_train_test = {"train": [], "test": []}

    # get X (subjects) and corresponding y (labels)
    if config.task == "AD" or config.task == "MCI":
        X = np.concatenate((partition_labels[config.class0], partition_labels[config.class1]), axis=0)
        y = np.array([0] * len(partition_labels[config.class0]) + [1] * len(partition_labels[config.class1]))
    elif config.task == "CN-MCI-AD":
        X = np.concatenate((partition_labels[config.class0], partition_labels[config.class1], partition_labels[config.class2]), axis=0)
        y = np.array([0] * len(partition_labels[config.class0]) + [1] * len(partition_labels[config.class1]) + [2] * len(partition_labels[config.class2]))

    # create k training and test sets (for stratified k cross validations)
    if config.shuffle_split:
        # random distribution of subjects over train and test sets
        skf = StratifiedShuffleSplit(n_splits=config.k_cross_validation, test_size=config.test_size, random_state=config.fixed_seed)
    else:
        # k folds: every subject in test set once
        skf = StratifiedKFold(n_splits=config.k_cross_validation, shuffle=True, random_state=config.fixed_seed)

    # split based on X and y
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        partition_train_test["train"].append(X_train)
        partition_train_test["test"].append(X_test)

    print("\nTRAIN TEST")
    count_sets(partition_train_test, labels)

    return partition_train_test, test_index


def split_train_val(partition_train_test, labels):
    """
    Splits a dataset in a training and validation set, based on stratified K folds

        INPUT:
            partition_train_test - train val should be extracted from training set only
            labels - see 'create_labels()'

        OUTPUT:
            partition_train_validation[k] - subjects split in train and val group
            format {"train" : [001, 002, 003], "validation" : [004, 005 ...] }
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    partition_train_validation = {"train": [], "validation": []}

    # for k-fold times
    for i in range(config.k_cross_validation):

        # regroup training set based on labels
        if config.task == "AD" or config.task == "MCI":
            temp = {0: [], 1: []}
        elif config.task == "CN-MCI-AD":
            temp = {0: [], 1: [], 2:[]}

        for id in partition_train_test["train"][i]:
            temp[labels[id]].append(id)

        # create X (subjects) and y (labels)
        if config.task == "AD" or config.task == "MCI":
            X = np.concatenate((temp[0], temp[1]), axis=0)
            y = np.array([0] * len(temp[0]) + [1] * len(temp[1]))
        elif config.task == "CN-MCI-AD":
            X = np.concatenate((temp[0], temp[1], temp[2]), axis=0)
            y = np.array([0] * len(temp[0]) + [1] * len(temp[1]) + [2] * len(temp[2]))

        # create stratified training and test set for that fold
        skf = StratifiedShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.fixed_seed)
        #skf = StratifiedKFold(n_splits=config.k_cross_validation, shuffle=True, random_state=config.fixed_seed)
        for train_index, validation_index in skf.split(X, y):
            X_train, X_validation = X[train_index], X[validation_index]
            partition_train_validation["train"].append(X_train)
            partition_train_validation["validation"].append(X_validation)

    print("\nTRAIN VALIDATION")
    count_sets(partition_train_validation, labels)

    return partition_train_validation




def format_labels(train_set, validation_set, test_set):
    """
    Splits a dataset in a training, validation, and test set

        INPUT:
            ...

        OUTPUT:
            partition_labels - format {"CN" : [001, 002, 003], "MCI" : [...], "AD" : [...] }
            labels - format {"001" : 0 , "002" : 1, "003" : 0 ...}
            partition_train_val_test - format {"train" : [001, 002, 003, ...], "val" : [...], "test" : [...] }
            ...
    """

    # create dicts
    partition_labels = {config.class0: [], config.class1: [], config.class2: []}
    labels = dict()
    partition_train_val_test = {"train": [], "validation": [], "test": []}

    for index, row in train_set.iterrows():
        if train_set.loc[index, "Diagnosis"] == 1.0:
            partition_labels[config.class0].append(train_set.loc[index, "PTID_VISCODE"])
            labels[train_set.loc[index, "PTID_VISCODE"]] = 0
        elif train_set.loc[index, "Diagnosis"] == 2.0:
            partition_labels[config.class1].append(train_set.loc[index, "PTID_VISCODE"])
            labels[train_set.loc[index, "PTID_VISCODE"]] = 1
        elif train_set.loc[index, "Diagnosis"] == 3.0:
            partition_labels[config.class2].append(train_set.loc[index, "PTID_VISCODE"])
            labels[train_set.loc[index, "PTID_VISCODE"]] = 2
        partition_train_val_test["train"].append(train_set.loc[index, "PTID_VISCODE"])

    for index, row in validation_set.iterrows():
        if validation_set.loc[index, "Diagnosis"] == 1.0:
            partition_labels[config.class0].append(validation_set.loc[index, "PTID_VISCODE"])
            labels[validation_set.loc[index, "PTID_VISCODE"]] = 0
        elif validation_set.loc[index, "Diagnosis"] == 2.0:
            partition_labels[config.class1].append(validation_set.loc[index, "PTID_VISCODE"])
            labels[validation_set.loc[index, "PTID_VISCODE"]] = 1
        elif validation_set.loc[index, "Diagnosis"] == 3.0:
            partition_labels[config.class2].append(validation_set.loc[index, "PTID_VISCODE"])
            labels[validation_set.loc[index, "PTID_VISCODE"]] = 2
        partition_train_val_test["validation"].append(validation_set.loc[index, "PTID_VISCODE"])

    for index, row in test_set.iterrows(): # different for TADPOLE?
        if test_set.loc[index, "Diagnosis"] == 1.0:
            partition_labels[config.class0].append(test_set.loc[index, "PTID_VISCODE"])
            labels[test_set.loc[index, "PTID_VISCODE"]] = 0
        elif test_set.loc[index, "Diagnosis"] == 2.0:
            partition_labels[config.class1].append(test_set.loc[index, "PTID_VISCODE"])
            labels[test_set.loc[index, "PTID_VISCODE"]] = 1
        elif test_set.loc[index, "Diagnosis"] == 3.0:
            partition_labels[config.class2].append(test_set.loc[index, "PTID_VISCODE"])
            labels[test_set.loc[index, "PTID_VISCODE"]] = 2
        partition_train_val_test["test"].append(test_set.loc[index, "PTID_VISCODE"])

    print("\nCATEGORIES\n")
    for cat in partition_labels:
        print("    " + cat + " : " + str(len(partition_labels[cat])))

    print("\nTRAIN VALIDATION TEST")

    # loop over set types (train/val/test)
    for set in partition_train_val_test:

        print("\n" + set)

        if not config.leaderboard and set == "test":
            print("The distribution of the test set for TADPOLE submissions is unknown.")
            continue

        a = []

        # loop over ids
        for id in partition_train_val_test[set]:
            a.append(labels[id])
        unique, counts = np.unique(a, return_counts=True)

        # replace with real labels
        c = []
        for u in unique:
            if u == 0:
                c.append(config.class0)
            elif u == 1:
                c.append(config.class1)
            elif u == 2:
                c.append(config.class2)

        # print distribution
        r = dict(zip(c, counts))
        print(r)


    #count_sets(partition_train_val_test, labels)

    return partition_labels, labels, partition_train_val_test



def create_sets_for_cv(df):
    """
        Splits a dataset in a training, validation, and test set

            INPUT:
                ...

            OUTPUT:
                partition_labels - format {"CN" : [001, 002, 003], "MCI" : [...], "AD" : [...] }
                labels - format {"001" : 0 , "002" : 1, "003" : 0 ...}
                partition_train_val_test - format {"train" : [001, 002, 003, ...], "val" : [...], "test" : [...] }
                ...
    """

    # create dicts
    if config.task == "AD" or config.task == "MCI":
        partition_labels = {config.class0: [], config.class1: []}
    else:
        partition_labels = {config.class0: [], config.class1: [], config.class2: []}
    labels_link = dict()
    labels_ptid_viscode = dict()

    for index, row in df.iterrows():
        if df.loc[index, "Diagnosis"] == 1.0:
            partition_labels[config.class0].append(df.loc[index, "Link"])
            labels_link[df.loc[index, "Link"]] = 0
            labels_ptid_viscode[df.loc[index, "PTID_VISCODE"]] = 0
        elif df.loc[index, "Diagnosis"] == 2.0:
            partition_labels[config.class1].append(df.loc[index, "Link"])
            labels_link[df.loc[index, "Link"]] = 1
            labels_ptid_viscode[df.loc[index, "PTID_VISCODE"]] = 1
        elif df.loc[index, "Diagnosis"] == 3.0:
            partition_labels[config.class2].append(df.loc[index, "Link"])
            labels_link[df.loc[index, "Link"]] = 2
            labels_ptid_viscode[df.loc[index, "PTID_VISCODE"]] = 2


    print("\nCATEGORIES\n")
    for cat in partition_labels:
        print("    " + cat + " : " + str(len(partition_labels[cat])))

    #-------------

    partition_train_test = {"train": [], "test": []}

    # get X (subjects) and corresponding y (labels)
    if config.task == "AD" or config.task == "MCI":
        X = np.concatenate((partition_labels[config.class0], partition_labels[config.class1]), axis=0)
        y = np.array([0] * len(partition_labels[config.class0]) + [1] * len(partition_labels[config.class1]))
    else:
        X = np.concatenate((partition_labels[config.class0], partition_labels[config.class1], partition_labels[config.class2]), axis=0)
        y = np.array([0] * len(partition_labels[config.class0]) + [1] * len(partition_labels[config.class1]) + [2] * len(partition_labels[config.class2]))

    rid_df = np.unique(df["RID"].copy())

    # create k training and test sets (for stratified k cross validations)
    if config.shuffle_split:
        # random distribution of subjects over train and test sets
        skf = ShuffleSplit(n_splits=config.k_cross_validation, test_size=config.test_size, random_state=config.fixed_seed)
    else:
        # k folds: every subject in test set once
        skf = KFold(n_splits=config.k_cross_validation, shuffle=True, random_state=config.fixed_seed)

    # split based on X
    for train_index, test_index in skf.split(rid_df):
        X_train_index = rid_df[train_index]
        X_test_index = rid_df[test_index]
        train_DF = df[df["RID"].isin(X_train_index)]
        train_link = train_DF["Link"].copy()
        mask_train = np.isin(X, train_link)
        test_DF = df[df["RID"].isin(X_test_index)]
        test_link = test_DF["Link"].copy()
        mask_test = np.isin(X, test_link)
        X_train = X[mask_train]
        X_test = X[mask_test]
        partition_train_test["train"].append(X_train)
        partition_train_test["test"].append(X_test)

    print("\nTRAIN TEST")
    count_sets(partition_train_test, labels_link)

    #-----------------

    partition_train_validation = {"train": [], "validation": []}

    # for k-fold times
    for i in range(config.k_cross_validation):

        # regroup training set based on labels
        # if config.task == "AD" or config.task == "MCI":
        #     temp = {0: [], 1: []}
        # elif config.task == "CN-MCI-AD":
        #     temp = {0: [], 1: [], 2: []}

        # for id in partition_train_test["train"][i]:
        #     temp[labels[id]].append(id)

        # # create X (subjects) and y (labels)
        # if config.task == "AD" or config.task == "MCI":
        #     X = np.concatenate((temp[0], temp[1]), axis=0)
        #     y = np.array([0] * len(temp[0]) + [1] * len(temp[1]))
        # else:
        #     X = np.concatenate((temp[0], temp[1], temp[2]), axis=0)
        #     y = np.array([0] * len(temp[0]) + [1] * len(temp[1]) + [2] * len(temp[2]))

        X = partition_train_test["train"][i]
        train_val_df = df[df["Link"].isin(X)]
        train_val_RIDs = np.unique(train_val_df["RID"].copy())

        # random distribution of subjects over train and validation set
        skf = ShuffleSplit(n_splits=1, test_size=config.val_size, random_state=config.fixed_seed)
        # skf = KFold(n_splits=config.k_cross_validation, shuffle=True, random_state=config.fixed_seed)

        for train_index, validation_index in skf.split(train_val_RIDs):
            X_train_index = train_val_RIDs[train_index]
            X_validation_index = train_val_RIDs[validation_index]
            train_DF = df[df["RID"].isin(X_train_index)]
            train_link = train_DF["Link"].copy()
            mask_train = np.isin(X, train_link)
            validation_DF = df[df["RID"].isin(X_validation_index)]
            validation_link = validation_DF["Link"].copy()
            mask_validation = np.isin(X, validation_link)
            X_train = X[mask_train]
            X_validation = X[mask_validation]
            partition_train_validation["train"].append(X_train)
            partition_train_validation["validation"].append(X_validation)

    print("\nTRAIN VALIDATION")
    count_sets(partition_train_validation, labels_link)

    return partition_labels, labels_link, labels_ptid_viscode, partition_train_test, partition_train_validation


def keep_one_mri_per_individual(train_test, train_validation, df, i):
    #np.random.seed(seed=config.fixed_seed)

    new_train_test = copy.deepcopy(train_test)
    new_train_validation = copy.deepcopy(train_validation)

    # for i in range(config.k_cross_validation):
    training_matches = new_train_validation["train"][i]
    validation_matches = new_train_validation["validation"][i]
    test_matches = new_train_test["test"][i]

    print("Amount of matches in training set for this interval: ", len(training_matches))
    print("Amount of matches in validation set for this interval: ", len(validation_matches))
    print("Amount of matches in test set for this interval: ", len(test_matches))
    print("Total amount of matches in this interval: ", df.shape)

    # For the test set, only keep one scan per individual
    new_test_df = df[df["Link"].isin(test_matches)]
    print("Amount of matches in test set for this interval: ", new_test_df.shape)
    unique_mri_test = new_test_df.drop_duplicates(subset="PTID_VISCODE")
    print("Amount of unique MRI scans in test set for this interval: ", unique_mri_test.shape)
    unique_rid_test = new_test_df.drop_duplicates(subset="RID")
    print("Amount of unique RIDs in test set for this interval: ", unique_rid_test.shape)
    #new_test_df = new_test_df[new_test_df['PTID_VISCODE'].str.endswith('_bl')]
    #print(new_test_df.shape)
    #new_test_df = new_test_df.groupby("RID").tail(1)
    #print(new_test_df.shape)
    size = 1  # sample size
    replace = True  # with replacement, doesn't matter if size = 1
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace), :]
    randomly_chosen_test_scans = new_test_df.groupby('RID', as_index=False).apply(fn)
    print("Amount of MRIs used for test set (chosen at random): ", randomly_chosen_test_scans.shape)

    test_link = randomly_chosen_test_scans["Link"].copy()
    #print(len(test_link))
    #print(len(new_train_test["test"][i]))
    mask_test = np.isin(new_train_test["test"][i], test_link)
    #print(len(mask_test))
    new_train_test["test"][i] = new_train_test["test"][i][mask_test]
    #print(len(new_train_test["test"][i]))

    # For the validation set, only keep one scan per individual
    new_validation_df = df[df["Link"].isin(validation_matches)]
    print("Amount of matches in validation set for this interval: ", new_validation_df.shape)
    unique_mri_validation = new_validation_df.drop_duplicates(subset="PTID_VISCODE")
    print("Amount of unique MRI scans in validation set for this interval: ", unique_mri_validation.shape)
    unique_rid_validation = new_validation_df.drop_duplicates(subset="RID")
    print("Amount of unique RIDs in validation set for this interval: ", unique_rid_validation.shape)
    randomly_chosen_validation_scans = new_validation_df.groupby('RID', as_index=False).apply(fn)
    print("Amount of MRIs used for validation set (chosen at random): ", randomly_chosen_validation_scans.shape)

    validation_link = randomly_chosen_validation_scans["Link"].copy()
    mask_validation = np.isin(new_train_validation["validation"][i], validation_link)
    new_train_validation["validation"][i] = new_train_validation["validation"][i][mask_validation]

    # For the training set, only keep one scan per individual
    new_training_df = df[df["Link"].isin(training_matches)]
    print("Amount of matches in training set for this interval: ", new_training_df.shape)
    unique_mri_training = new_training_df.drop_duplicates(subset="PTID_VISCODE")
    print("Amount of unique MRI scans in training set for this interval: ", unique_mri_training.shape)
    unique_rid_training = new_training_df.drop_duplicates(subset="RID")
    print("Amount of unique RIDs in training set for this interval: ", unique_rid_training.shape)
    randomly_chosen_training_scans = new_training_df.groupby('RID', as_index=False).apply(fn)
    print("Amount of MRIs used for training set (chosen at random): ", randomly_chosen_training_scans.shape)

    training_link = randomly_chosen_training_scans["Link"].copy()
    mask_training = np.isin(new_train_validation["train"][i], training_link)
    new_train_validation["train"][i] = new_train_validation["train"][i][mask_training]

    # Create a copy that contains the ID's of unique MRI scans ("PTID_VISCODE") rather than the matches ("Link")
    train_validation_ptid_viscode = copy.deepcopy(new_train_validation)

    #print(len(train_validation_ptid_viscode["train"][i]))
    temp_DF = df[df["Link"].isin(train_validation_ptid_viscode["train"][i])]
    temp_PTID_VISCODE = temp_DF["PTID_VISCODE"].copy()
    train_validation_ptid_viscode["train"][i] = np.asarray(temp_PTID_VISCODE)
    #print(len(train_validation_ptid_viscode["train"][i]))

    #print(len(train_validation_ptid_viscode["validation"][i]))
    temp_DF_val = df[df["Link"].isin(train_validation_ptid_viscode["validation"][i])]
    temp_PTID_VISCODE_val = temp_DF_val["PTID_VISCODE"].copy()
    train_validation_ptid_viscode["validation"][i] = np.asarray(temp_PTID_VISCODE_val)
    #print(len(train_validation_ptid_viscode["validation"][i]))

    train_test_ptid_viscode = copy.deepcopy(new_train_test)

    #print(len(train_test_ptid_viscode["test"][i]))
    temp_DF_test = df[df["Link"].isin(train_test_ptid_viscode["test"][i])]
    temp_PTID_VISCODE_test = temp_DF_test["PTID_VISCODE"].copy()
    train_test_ptid_viscode["test"][i] = np.asarray(temp_PTID_VISCODE_test)
    #print(len(train_test_ptid_viscode["test"][i]))

    return new_train_test, new_train_validation, train_validation_ptid_viscode, train_test_ptid_viscode


def non_imaging_sets(partition_train_validation, partition_train_test, all_features, i):
    train_features = all_features[all_features["Link"].isin(partition_train_validation["train"][i])]
    validation_features = all_features[all_features["Link"].isin(partition_train_validation["validation"][i])]
    test_features = all_features[all_features["Link"].isin(partition_train_test["test"][i])]

    return train_features, validation_features, test_features


def count_sets(dic, labels):
    """
    Counts and prints the amount of sets present per k-fold and class

        INPUT:
            dic - dictionary of type dic["set"]["k-fold"]["id"]
            labels - dictionary of type labels["id"]["class"]

        OUTPUT:
            print overview of set distributions
    """

    # loop over set types (train/test)
    for set in dic:
        print("\n" + set)

        # loop over k-folds
        for i in range(len(dic[set])):
            a = []

            # loop over ids
            for id in dic[set][i]:
                a.append(labels[id])
            unique, counts = np.unique(a, return_counts=True)

            # replace with real labels
            c = []
            for u in unique:
                if u == 0:
                    c.append(config.class0)
                elif u == 1:
                    c.append(config.class1)
                elif u == 2:
                    c.append(config.class2)

            # print distribution
            r = dict(zip(c, counts))
            print("    fold " + str(i) + ":", r)
