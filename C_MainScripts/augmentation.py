# import configuration file
import config

# set random seed
#from numpy.random import seed
#from tensorflow import set_random_seed
#seed(config.fixed_seed)
#set_random_seed(config.fixed_seed)

import os
import math
import numpy as np
import h5py


def augmentation(dataset, labels):
    """
    Applies augmentation on a given dataset and appends the new images to the original dataset and labels

        INPUT:
            dataset, labels - original dataset and labels on which augmentation should be performed

        OUTPUT:
            dataset, labels - new sets including augmented images
            saves the augmented images in the given directory
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    print("Augmentation")

    # if necessary create aug dir and make sure it's empty
    if not os.path.exists(config.aug_dir):
        os.makedirs(config.aug_dir)
    else:
        os.system('rm -rf %s/*' % config.aug_dir)

    # sort ids based on category
    if config.task == "AD" or config.task == "MCI":
        split_categories = {0: [], 1: []}
    else:
        split_categories = {0: [], 1: [], 2: []}
    for id in dataset:
        split_categories[labels[id]].append(id) # change!

    # calculate the amount of missing images to be augmented
    if config.task == "AD" or config.task == "MCI":
        missing = {0: max(0, config.class_total - len(split_categories[0])), 1: max(0, config.class_total - len(split_categories[1]))}
        print("    missing " + config.class0 + " data: ", missing[0])
        print("    missing " + config.class1 + " data: ", missing[1])
    else:
        missing = {0: max(0, config.class_total - len(split_categories[0])), 1: max(0, config.class_total - len(split_categories[1])), 2: max(0, config.class_total - len(split_categories[2]))}
        print("    missing " + config.class0 + " data: ", missing[0])
        print("    missing " + config.class1 + " data: ", missing[1])
        print("    missing " + config.class2 + " data: ", missing[2])

    cnt = 0

    # loop over categories
    for cat in split_categories:

        # loop over missing repetitions of whole dataset
        for rep_idx in range(math.floor(missing[cat] / len(split_categories[cat]))):

            # loop over ids in dataset
            for id in split_categories[cat]:

                aug_name = "aug" + str(cnt) + "_" + id

                # update labels + dataset
                labels[aug_name] = cat
                dataset = np.append(dataset, aug_name)

                # augment image + save
                aug_image = mixing(id, split_categories[cat])
                if config.task == "AD" or config.task == "MCI":
                    np.save(config.aug_dir + aug_name + ".npy", aug_image)
                else:
                    with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
                        hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)

                cnt += 1

        # loop over rest of the missing images
        for rest_idx in range(missing[cat] % len(split_categories[cat])):

            id = split_categories[cat][rest_idx]
            aug_name = "aug" + str(cnt) + "_" + id

            # update labels + dataset
            labels[aug_name] = cat
            dataset = np.append(dataset, aug_name)

            # augment image + save
            aug_image = mixing(id, split_categories[cat])
            if config.task == "AD" or config.task == "MCI":
                np.save(config.aug_dir + aug_name + ".npy", aug_image)
            else:
                with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
                    hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)

            cnt += 1

    return dataset, labels


def mixing(id, list_ids):
    """
    Applies augmentation on an image by mixing the original image with a random image of the same category.
    The augmentation factor indicates the percentage / fraction of the random image which will be used.

        INPUT:
            id - id of subject to be augmented
            list_ids - list with all ids to pick mix image

        OUTPUT:
            aug_image - the augmented image
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)
    if config.task == "AD" or config.task == "MCI":
        # load original image
        image = np.load(config.data_dir + id + ".npy")

        # load random image from same category
        id_mix = np.random.choice(np.setdiff1d(list_ids, [id]))
        image_mix = np.load(config.data_dir + id_mix + ".npy")

        # mix images
        aug_image = (1 - config.aug_factor) * image + config.aug_factor * image_mix
    else:
        # load original image
        with h5py.File(config.data_dir + id + '.h5py', 'r') as hf:
            image = hf[id][:]

        # load random image from same category
        id_mix = np.random.choice(np.setdiff1d(list_ids, [id]))
        with h5py.File(config.data_dir + id_mix + '.h5py', 'r') as hf:
            image_mix = hf[id_mix][:]

        # mix images
        aug_image = (1 - config.aug_factor) * image + config.aug_factor * image_mix

    return aug_image


def augmentation_mix_all(dataset, dataset_links, labels, train_features, all_data):
    """
    Applies augmentation on a given dataset and appends the new images to the original dataset and labels

        INPUT:
            dataset, labels - original dataset and labels on which augmentation should be performed

        OUTPUT:
            dataset, labels - new sets including augmented images
            saves the augmented images in the given directory
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    print("Augmentation")

    # if necessary create aug dir and make sure it's empty
    if not os.path.exists(config.aug_dir):
        os.makedirs(config.aug_dir)
    else:
        os.system('rm -rf %s/*' % config.aug_dir)

    # sort ids based on category
    if config.task == "AD" or config.task == "MCI":
        split_categories = {0: [], 1: []}
    else:
        split_categories = {0: [], 1: [], 2: []}
    #     if config.augment_converters_separately == True:
    #         split_categories_converters = {0: [], 1: [], 2: []}
    #         split_categories_non_converters = {0: [], 1: [], 2: []}
    # if config.augment_converters_separately == False:
    for id in dataset:
        split_categories[labels[id]].append(id)
    # else:
    #     for id in dataset:
    #         all_links = all_data[all_data["PTID_VISCODE"] == id]
    #         this_link = all_links[all_links["Link"].isin(dataset_links)]
    #         index = this_link.index.item()
    #         if this_link.loc[index, "Converter"] == True:
    #             split_categories_converters[labels[id]].append(id)
    #         else:
    #             split_categories_non_converters[labels[id]].append(id)

    # calculate the amount of missing images to be augmented
    if config.task == "AD" or config.task == "MCI":
        missing = {0: max(0, config.class_total - len(split_categories[0])), 1: max(0, config.class_total - len(split_categories[1]))}
        print("    missing " + config.class0 + " data: ", missing[0])
        print("    missing " + config.class1 + " data: ", missing[1])
    else:
        missing = {0: max(0, config.class_total - len(split_categories[0])), 1: max(0, config.class_total - len(split_categories[1])), 2: max(0, config.class_total - len(split_categories[2]))}
        print("    missing " + config.class0 + " data: ", missing[0])
        print("    missing " + config.class1 + " data: ", missing[1])
        print("    missing " + config.class2 + " data: ", missing[2])

    cnt = 0

    # if config.augment_converters_separately == True:
    #     missing_converters = {0: config.converters_percentage * missing[0], 1: config.converters_percentage * missing[1], 2: config.converters_percentage * missing[2]}
    #     print("    missing converters " + config.class0 + " data: ", missing_converters[0])
    #     print("    missing converters " + config.class1 + " data: ", missing_converters[1])
    #     print("    missing converters " + config.class2 + " data: ", missing_converters[2])
    #     missing_non_converters = {0: (1-config.converters_percentage) * missing[0], 1: (1-config.converters_percentage) * missing[1], 2: (1-config.converters_percentage) * missing[2]}
    #     print("    missing non converters " + config.class0 + " data: ", missing_non_converters[0])
    #     print("    missing non converters " + config.class1 + " data: ", missing_non_converters[1])
    #     print("    missing non converters " + config.class2 + " data: ", missing_non_converters[2])

    # loop over categories
    for cat in split_categories:

        image_shape = config.input_shape

        # if config.augment_converters_separately == False:

        number_of_images = len(split_categories[cat])

        # Load all images with this label
        image_dict = dict.fromkeys(split_categories[cat])

        for existing_image in split_categories[cat]:
            with h5py.File(config.data_dir + existing_image + '.h5py', 'r') as hf:
                image = hf[existing_image][:]

            image_dict[existing_image] = image

        for augmented in range(missing[cat]):

            aug_name = "aug" + str(cnt)

            # update labels + dataset
            labels[aug_name] = cat
            dataset = np.append(dataset, aug_name)

            # initiate new image
            aug_image = np.zeros(shape=image_shape)

            # create new image based on dirichlet distribution of all available images
            vector_of_ones = [config.dirichlet_alpha] * number_of_images
            dirichlet_weights = np.random.dirichlet(alpha=vector_of_ones, size=1)
            #print("sum: ", np.sum(dirichlet_weights))
            #print("min: ", np.min(dirichlet_weights))
            #print("max: ", np.max(dirichlet_weights))

            if config.non_image_data == True:
                new_row = train_features.iloc[0, :].copy()
                new_row["RID"] = aug_name
                new_row["PTID_VISCODE"] = aug_name
                new_row["Link"] = aug_name
                new_row["MMSE"] = 0
                new_row["ADAS13"] = 0
                new_row["Current_diagnosis"] = 0
                new_row["Time_feature"] = 0

            for idx, image_id in enumerate(split_categories[cat]):
                aug_image += dirichlet_weights[0][idx] * image_dict[image_id]
                if config.non_image_data == True:
                    this_row = train_features[train_features["PTID_VISCODE"] == image_id].copy()
                    this_row = this_row[this_row["Link"].isin(dataset_links)]
                    index = this_row.index.item()
                    new_row["MMSE"] += dirichlet_weights[0][idx] * this_row.loc[index, "MMSE"]
                    new_row["ADAS13"] += dirichlet_weights[0][idx] * this_row.loc[index, "ADAS13"]
                    new_row["Current_diagnosis"] += dirichlet_weights[0][idx] * this_row.loc[index, "Current_diagnosis"]
                    new_row["Time_feature"] += dirichlet_weights[0][idx] * this_row.loc[index, "Time_feature"]

            if config.non_image_data == True:
                train_features = train_features.append(new_row)

            # store new image in augmentation output directory
            if config.task == "AD" or config.task == "MCI":
                print("This doesn't work yet for task AD or task MCI")
            else:
                with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
                    hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)

            cnt += 1

        # else:
        #     # augmentation converters
        #     number_of_images_converters = len(split_categories_converters[cat])
        #
        #     # Load all images with this label
        #     image_dict_converters = dict.fromkeys(split_categories_converters[cat])
        #
        #     for existing_image in split_categories_converters[cat]:
        #         with h5py.File(config.data_dir + existing_image + '.h5py', 'r') as hf:
        #             image = hf[existing_image][:]
        #
        #         image_dict_converters[existing_image] = image
        #
        #     for augmented in range(missing_converters[cat]):
        #
        #         aug_name = "aug" + str(cnt)
        #
        #         # update labels + dataset
        #         labels[aug_name] = cat
        #         dataset = np.append(dataset, aug_name)
        #
        #         # initiate new image
        #         aug_image = np.zeros(shape=image_shape)
        #
        #         # create new image based on dirichlet distribution of all available images
        #         vector_of_ones = [config.dirichlet_alpha] * number_of_images_converters
        #         dirichlet_weights = np.random.dirichlet(alpha=vector_of_ones, size=1)
        #         # print("sum: ", np.sum(dirichlet_weights))
        #         # print("min: ", np.min(dirichlet_weights))
        #         # print("max: ", np.max(dirichlet_weights))
        #
        #         if config.non_image_data == True:
        #             new_row = train_features.iloc[0, :].copy()
        #             new_row["RID"] = aug_name
        #             new_row["PTID_VISCODE"] = aug_name
        #             new_row["Link"] = aug_name
        #             new_row["MMSE"] = 0
        #             new_row["ADAS13"] = 0
        #             new_row["Current_diagnosis"] = 0
        #             new_row["Time_feature"] = 0
        #
        #         for idx, image_id in enumerate(split_categories_converters[cat]):
        #             aug_image += dirichlet_weights[0][idx] * image_dict[image_id]
        #             if config.non_image_data == True:
        #                 this_row = train_features[train_features["PTID_VISCODE"] == image_id].copy()
        #                 this_row = this_row[this_row["Link"].isin(dataset_links)]
        #                 index = this_row.index.item()
        #                 new_row["MMSE"] += dirichlet_weights[0][idx] * this_row.loc[index, "MMSE"]
        #                 new_row["ADAS13"] += dirichlet_weights[0][idx] * this_row.loc[index, "ADAS13"]
        #                 new_row["Current_diagnosis"] += dirichlet_weights[0][idx] * this_row.loc[index, "Current_diagnosis"]
        #                 new_row["Time_feature"] += dirichlet_weights[0][idx] * this_row.loc[index, "Time_feature"]
        #
        #         if config.non_image_data == True:
        #             train_features = train_features.append(new_row)
        #
        #         # store new image in augmentation output directory
        #         if config.task == "AD" or config.task == "MCI":
        #             print("This doesn't work yet for task AD or task MCI")
        #         else:
        #             with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
        #                 hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)
        #
        #         cnt += 1
        #
        #
        #     # augmentation non-converters
        #     number_of_images_non_converters = len(split_categories_non_converters[cat])
        #
        #     # Load all images with this label
        #     image_dict_non_converters = dict.fromkeys(split_categories_non_converters[cat])
        #
        #     for existing_image in split_categories_non_converters[cat]:
        #         with h5py.File(config.data_dir + existing_image + '.h5py', 'r') as hf:
        #             image = hf[existing_image][:]
        #
        #         image_dict_non_converters[existing_image] = image
        #
        #     for augmented in range(missing_non_converters[cat]):
        #
        #         aug_name = "aug" + str(cnt)
        #
        #         # update labels + dataset
        #         labels[aug_name] = cat
        #         dataset = np.append(dataset, aug_name)
        #
        #         # initiate new image
        #         aug_image = np.zeros(shape=image_shape)
        #
        #         # create new image based on dirichlet distribution of all available images
        #         vector_of_ones = [config.dirichlet_alpha] * number_of_images_non_converters
        #         dirichlet_weights = np.random.dirichlet(alpha=vector_of_ones, size=1)
        #         # print("sum: ", np.sum(dirichlet_weights))
        #         # print("min: ", np.min(dirichlet_weights))
        #         # print("max: ", np.max(dirichlet_weights))
        #
        #         if config.non_image_data == True:
        #             new_row = train_features.iloc[0, :].copy()
        #             new_row["RID"] = aug_name
        #             new_row["PTID_VISCODE"] = aug_name
        #             new_row["Link"] = aug_name
        #             new_row["MMSE"] = 0
        #             new_row["ADAS13"] = 0
        #             new_row["Current_diagnosis"] = 0
        #             new_row["Time_feature"] = 0
        #
        #         for idx, image_id in enumerate(split_categories_non_converters[cat]):
        #             aug_image += dirichlet_weights[0][idx] * image_dict[image_id]
        #             if config.non_image_data == True:
        #                 this_row = train_features[train_features["PTID_VISCODE"] == image_id].copy()
        #                 this_row = this_row[this_row["Link"].isin(dataset_links)]
        #                 index = this_row.index.item()
        #                 new_row["MMSE"] += dirichlet_weights[0][idx] * this_row.loc[index, "MMSE"]
        #                 new_row["ADAS13"] += dirichlet_weights[0][idx] * this_row.loc[index, "ADAS13"]
        #                 new_row["Current_diagnosis"] += dirichlet_weights[0][idx] * this_row.loc[index, "Current_diagnosis"]
        #                 new_row["Time_feature"] += dirichlet_weights[0][idx] * this_row.loc[index, "Time_feature"]
        #
        #         if config.non_image_data == True:
        #             train_features = train_features.append(new_row)
        #
        #         # store new image in augmentation output directory
        #         if config.task == "AD" or config.task == "MCI":
        #             print("This doesn't work yet for task AD or task MCI")
        #         else:
        #             with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
        #                 hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)
        #
        #         cnt += 1


    return dataset, labels, train_features


def augmentation_mix_only_converters(dataset, dataset_links, labels, train_features, all_data):
    """
    Applies augmentation on a given dataset and appends the new images to the original dataset and labels

        INPUT:
            dataset, labels - original dataset and labels on which augmentation should be performed

        OUTPUT:
            dataset, labels - new sets including augmented images
            saves the augmented images in the given directory
    """

    # set seeds
    #seed(config.fixed_seed)
    #set_random_seed(config.fixed_seed)

    print("Augmentation")

    # if necessary create aug dir and make sure it's empty
    if not os.path.exists(config.aug_dir):
        os.makedirs(config.aug_dir)
    else:
        os.system('rm -rf %s/*' % config.aug_dir)

    # sort ids based on category
    if config.task == "AD" or config.task == "MCI":
        split_categories = {0: [], 1: []}
    else:
        split_categories = {0: [], 1: [], 2: []}
        #split_categories = {1: [], 2: []}

    #     if config.augment_converters_separately == True:
    #         split_categories_converters = {0: [], 1: [], 2: []}
    #         split_categories_non_converters = {0: [], 1: [], 2: []}
    # if config.augment_converters_separately == False:

    # for id in dataset:
    #     split_categories[labels[id]].append(id)

    # else:
    #     for id in dataset:
    #         all_links = all_data[all_data["PTID_VISCODE"] == id]
    #         this_link = all_links[all_links["Link"].isin(dataset_links)]
    #         index = this_link.index.item()
    #         if this_link.loc[index, "Converter"] == True:
    #             split_categories_converters[labels[id]].append(id)
    #         else:
    #             split_categories_non_converters[labels[id]].append(id)

    # calculate the amount of missing images to be augmented
    total_train_samples = dataset.shape
    print("Total train samples: ", total_train_samples)

    #test = all_data["PTID_VISCODE"].isin(dataset)

    converters = all_data[(all_data["PTID_VISCODE"].isin(dataset)) & (all_data["Converter"] == True)]
    unique_converters = converters.drop_duplicates("PTID_VISCODE", keep="last")
    total_converters = unique_converters.shape[0]
    print("Total converters: ", total_converters)
    non_converters = all_data[(all_data["PTID_VISCODE"].isin(dataset)) & (all_data["Converter"] == False)]
    unique_non_converters = non_converters.drop_duplicates("PTID_VISCODE", keep="last")
    total_non_converters = unique_non_converters.shape[0]
    print("Total non converters: ", total_non_converters)

    total_to_augment = total_non_converters - total_converters

    converters_to_CN = unique_converters[unique_converters["Diagnosis"] == 1]
    converters_to_CN_count = converters_to_CN.shape[0]
    print("Total converters to CN: ", converters_to_CN_count)       # should be zero
    converters_to_MCI = unique_converters[unique_converters["Diagnosis"] == 2]
    converters_to_MCI_count = converters_to_MCI.shape[0]
    print("Total converters to MCI: ", converters_to_MCI_count)
    converters_to_AD = unique_converters[unique_converters["Diagnosis"] == 3]
    converters_to_AD_count = converters_to_AD.shape[0]
    print("Total converters to AD: ", converters_to_AD_count)

    ratio_MCI = converters_to_MCI_count / total_converters
    ratio_AD = converters_to_AD_count / total_converters

    MCI_to_augment = round(ratio_MCI * total_to_augment)
    #AD_to_augment = ratio_AD * total_to_augment
    AD_to_augment = total_to_augment - MCI_to_augment

    unique_converter_list = unique_converters["PTID_VISCODE"].copy()
    for id in unique_converter_list:
        split_categories[labels[id]].append(id)

    if config.task == "AD" or config.task == "MCI":
        missing = {0: max(0, config.class_total - len(split_categories[0])), 1: max(0, config.class_total - len(split_categories[1]))}
        # print("    missing " + config.class0 + " data: ", missing[0])
        # print("    missing " + config.class1 + " data: ", missing[1])
    else:
        # missing = {0: max(0, config.class_total - len(split_categories[0])), 1: max(0, config.class_total - len(split_categories[1])), 2: max(0, config.class_total - len(split_categories[2]))}
        missing = {0: 0, 1: MCI_to_augment, 2: AD_to_augment}
        # print("    missing " + config.class0 + " data: ", missing[0])
        # print("    missing " + config.class1 + " data: ", missing[1])
        # print("    missing " + config.class2 + " data: ", missing[2])
        print("    missing " + config.class0 + " data: ", missing[0])
        print("    missing " + config.class1 + " data: ", missing[1])
        print("    missing " + config.class2 + " data: ", missing[2])

    cnt = 0

    # if config.augment_converters_separately == True:
    #     missing_converters = {0: config.converters_percentage * missing[0], 1: config.converters_percentage * missing[1], 2: config.converters_percentage * missing[2]}
    #     print("    missing converters " + config.class0 + " data: ", missing_converters[0])
    #     print("    missing converters " + config.class1 + " data: ", missing_converters[1])
    #     print("    missing converters " + config.class2 + " data: ", missing_converters[2])
    #     missing_non_converters = {0: (1-config.converters_percentage) * missing[0], 1: (1-config.converters_percentage) * missing[1], 2: (1-config.converters_percentage) * missing[2]}
    #     print("    missing non converters " + config.class0 + " data: ", missing_non_converters[0])
    #     print("    missing non converters " + config.class1 + " data: ", missing_non_converters[1])
    #     print("    missing non converters " + config.class2 + " data: ", missing_non_converters[2])

    # loop over categories
    for cat in split_categories:

        image_shape = config.input_shape

        # if config.augment_converters_separately == False:

        number_of_images = len(split_categories[cat])

        # Load all images with this label
        image_dict = dict.fromkeys(split_categories[cat])

        for existing_image in split_categories[cat]:
            with h5py.File(config.data_dir + existing_image + '.h5py', 'r') as hf:
                image = hf[existing_image][:]

            image_dict[existing_image] = image

        for augmented in range(missing[cat]):

            aug_name = "aug" + str(cnt)

            # update labels + dataset
            labels[aug_name] = cat
            dataset = np.append(dataset, aug_name)

            # initiate new image
            aug_image = np.zeros(shape=image_shape)

            # create new image based on dirichlet distribution of all available images
            config.dirichlet_alpha = 1/number_of_images
            vector_of_ones = [config.dirichlet_alpha] * number_of_images
            dirichlet_weights = np.random.dirichlet(alpha=vector_of_ones, size=1)
            print("sum: ", np.sum(dirichlet_weights))
            print("min: ", np.min(dirichlet_weights))
            print("max: ", np.max(dirichlet_weights))

            if config.non_image_data == True:
                new_row = train_features.iloc[0, :].copy()
                new_row["RID"] = aug_name
                new_row["PTID_VISCODE"] = aug_name
                new_row["Link"] = aug_name
                new_row["MMSE"] = 0
                new_row["ADAS13"] = 0
                new_row["Current_diagnosis"] = 0
                new_row["Time_feature"] = 0
                new_row["Converter"] = True

            for idx, image_id in enumerate(split_categories[cat]):
                aug_image += dirichlet_weights[0][idx] * image_dict[image_id]
                if config.non_image_data == True:
                    this_row = train_features[train_features["PTID_VISCODE"] == image_id].copy()
                    this_row = this_row[this_row["Link"].isin(dataset_links)]
                    index = this_row.index.item()
                    new_row["MMSE"] += dirichlet_weights[0][idx] * this_row.loc[index, "MMSE"]
                    new_row["ADAS13"] += dirichlet_weights[0][idx] * this_row.loc[index, "ADAS13"]
                    new_row["Current_diagnosis"] += dirichlet_weights[0][idx] * this_row.loc[index, "Current_diagnosis"]
                    new_row["Time_feature"] += dirichlet_weights[0][idx] * this_row.loc[index, "Time_feature"]

            if config.non_image_data == True:
                train_features = train_features.append(new_row)

            # store new image in augmentation output directory
            if config.task == "AD" or config.task == "MCI":
                print("This doesn't work yet for task AD or task MCI")
            else:
                with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
                    hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)

            cnt += 1

        # else:
        #     # augmentation converters
        #     number_of_images_converters = len(split_categories_converters[cat])
        #
        #     # Load all images with this label
        #     image_dict_converters = dict.fromkeys(split_categories_converters[cat])
        #
        #     for existing_image in split_categories_converters[cat]:
        #         with h5py.File(config.data_dir + existing_image + '.h5py', 'r') as hf:
        #             image = hf[existing_image][:]
        #
        #         image_dict_converters[existing_image] = image
        #
        #     for augmented in range(missing_converters[cat]):
        #
        #         aug_name = "aug" + str(cnt)
        #
        #         # update labels + dataset
        #         labels[aug_name] = cat
        #         dataset = np.append(dataset, aug_name)
        #
        #         # initiate new image
        #         aug_image = np.zeros(shape=image_shape)
        #
        #         # create new image based on dirichlet distribution of all available images
        #         vector_of_ones = [config.dirichlet_alpha] * number_of_images_converters
        #         dirichlet_weights = np.random.dirichlet(alpha=vector_of_ones, size=1)
        #         # print("sum: ", np.sum(dirichlet_weights))
        #         # print("min: ", np.min(dirichlet_weights))
        #         # print("max: ", np.max(dirichlet_weights))
        #
        #         if config.non_image_data == True:
        #             new_row = train_features.iloc[0, :].copy()
        #             new_row["RID"] = aug_name
        #             new_row["PTID_VISCODE"] = aug_name
        #             new_row["Link"] = aug_name
        #             new_row["MMSE"] = 0
        #             new_row["ADAS13"] = 0
        #             new_row["Current_diagnosis"] = 0
        #             new_row["Time_feature"] = 0
        #
        #         for idx, image_id in enumerate(split_categories_converters[cat]):
        #             aug_image += dirichlet_weights[0][idx] * image_dict[image_id]
        #             if config.non_image_data == True:
        #                 this_row = train_features[train_features["PTID_VISCODE"] == image_id].copy()
        #                 this_row = this_row[this_row["Link"].isin(dataset_links)]
        #                 index = this_row.index.item()
        #                 new_row["MMSE"] += dirichlet_weights[0][idx] * this_row.loc[index, "MMSE"]
        #                 new_row["ADAS13"] += dirichlet_weights[0][idx] * this_row.loc[index, "ADAS13"]
        #                 new_row["Current_diagnosis"] += dirichlet_weights[0][idx] * this_row.loc[index, "Current_diagnosis"]
        #                 new_row["Time_feature"] += dirichlet_weights[0][idx] * this_row.loc[index, "Time_feature"]
        #
        #         if config.non_image_data == True:
        #             train_features = train_features.append(new_row)
        #
        #         # store new image in augmentation output directory
        #         if config.task == "AD" or config.task == "MCI":
        #             print("This doesn't work yet for task AD or task MCI")
        #         else:
        #             with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
        #                 hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)
        #
        #         cnt += 1
        #
        #
        #     # augmentation non-converters
        #     number_of_images_non_converters = len(split_categories_non_converters[cat])
        #
        #     # Load all images with this label
        #     image_dict_non_converters = dict.fromkeys(split_categories_non_converters[cat])
        #
        #     for existing_image in split_categories_non_converters[cat]:
        #         with h5py.File(config.data_dir + existing_image + '.h5py', 'r') as hf:
        #             image = hf[existing_image][:]
        #
        #         image_dict_non_converters[existing_image] = image
        #
        #     for augmented in range(missing_non_converters[cat]):
        #
        #         aug_name = "aug" + str(cnt)
        #
        #         # update labels + dataset
        #         labels[aug_name] = cat
        #         dataset = np.append(dataset, aug_name)
        #
        #         # initiate new image
        #         aug_image = np.zeros(shape=image_shape)
        #
        #         # create new image based on dirichlet distribution of all available images
        #         vector_of_ones = [config.dirichlet_alpha] * number_of_images_non_converters
        #         dirichlet_weights = np.random.dirichlet(alpha=vector_of_ones, size=1)
        #         # print("sum: ", np.sum(dirichlet_weights))
        #         # print("min: ", np.min(dirichlet_weights))
        #         # print("max: ", np.max(dirichlet_weights))
        #
        #         if config.non_image_data == True:
        #             new_row = train_features.iloc[0, :].copy()
        #             new_row["RID"] = aug_name
        #             new_row["PTID_VISCODE"] = aug_name
        #             new_row["Link"] = aug_name
        #             new_row["MMSE"] = 0
        #             new_row["ADAS13"] = 0
        #             new_row["Current_diagnosis"] = 0
        #             new_row["Time_feature"] = 0
        #
        #         for idx, image_id in enumerate(split_categories_non_converters[cat]):
        #             aug_image += dirichlet_weights[0][idx] * image_dict[image_id]
        #             if config.non_image_data == True:
        #                 this_row = train_features[train_features["PTID_VISCODE"] == image_id].copy()
        #                 this_row = this_row[this_row["Link"].isin(dataset_links)]
        #                 index = this_row.index.item()
        #                 new_row["MMSE"] += dirichlet_weights[0][idx] * this_row.loc[index, "MMSE"]
        #                 new_row["ADAS13"] += dirichlet_weights[0][idx] * this_row.loc[index, "ADAS13"]
        #                 new_row["Current_diagnosis"] += dirichlet_weights[0][idx] * this_row.loc[index, "Current_diagnosis"]
        #                 new_row["Time_feature"] += dirichlet_weights[0][idx] * this_row.loc[index, "Time_feature"]
        #
        #         if config.non_image_data == True:
        #             train_features = train_features.append(new_row)
        #
        #         # store new image in augmentation output directory
        #         if config.task == "AD" or config.task == "MCI":
        #             print("This doesn't work yet for task AD or task MCI")
        #         else:
        #             with h5py.File(config.aug_dir + aug_name + ".h5py", "w") as hf:
        #                 hf.create_dataset(aug_name, data=aug_image, compression="gzip", compression_opts=9)
        #
        #         cnt += 1


    return dataset, labels, train_features