import os
import sys
from shutil import copyfile
import h5py

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.engine.saving import load_model
from skimage.transform import resize

# import configuration parameters
from vis_config import roi, task, label, classification_type, data_set, gc_layer, server, class_limit, model_id, interval, repetition_id, non_image_features, learning_phase, run    #, gb_layer


def main():
    """
    The current script implements a gradCAM analysis of an AD classification model.
    In the configuration file 'vis_config.py' the settings can be specified.

    This script computes the Grad-CAM mean and variation of a batch of images. Since in most
    cases memory is restricted to calculating the Grad-CAM of only 10 subjects in one run, this script
    is designed to be run several times after each other until a Grad-CAM of all subjects is calculated.
    For this reason this script uses an input argument which defines the run, and can based on this select
    the 10 subjects which belong in that run.

    After running this script the Grad-CAMs created in each run can be calculated using the script
    'vis_average.py'. Here a final nifti version of both the mean and variation will be created.

    The gradcam computations are adapted from the code on:
    https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py
    https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py

    Implementation is based on the paper of Selvaraju et al. (2017):
    https://arxiv.org/pdf/1610.02391.pdf
    """

    # set info paths
    if server:
        misclassification_path = f"/path/to/D_OutputFiles/F_GradCAM_output/A_Misclassifications/{model_id}_k{repetition_id}/"
        #run = int(sys.argv[3])
    else:
        misclassification_path = f"/path/to/D_OutputFiles/F_GradCAM_output/A_Misclassifications/{model_id}_k{repetition_id}/"

    # load subject file
    all_subjects = np.load(f"{misclassification_path}{classification_type}_classified_subjects_{label}_{data_set}.npy", allow_pickle=True)
    #k_splits = all_subjects.shape[0]

    # set paths (when running on server based on data dir + job nr input)
    if server:
        data_path = sys.argv[1] + "/"
        info_path = f"/path/to/D_OutputFiles/B_Output/{model_id}/"
        features = f"/path/to/A_DataFiles/PreprocessedData/newPatients/{interval}/all_non_image_features.csv"
        save_path = f"/path/to/D_OutputFiles/F_GradCAM_output/B_Main/{model_id}_k{repetition_id}/"
        run_path = f"{save_path}{label}_{data_set}/run{sys.argv[3]}_{roi}_n{class_limit}_gc{gc_layer}_{label}/"
    else:
        data_path = f"/path/to/A_DataFiles/data_ADNI_GM_f4/"
        info_path = f"/path/to/D_OutputFiles/B_Output/{model_id}/"
        features = f"/path/to/A_DataFiles/PreprocessedData/newPatients/{interval}/all_non_image_features.csv"
        save_path = f"/path/to/D_OutputFiles/F_GradCAM_output/B_Main/{model_id}_k{repetition_id}/"
        run_path = f"{save_path}{label}_{data_set}/run{run}_{roi}_n{class_limit}_gc{gc_layer}_{label}/"

    create_data_directory(run_path)
    all_features = pd.read_csv(features)

    if run == 0:
        if server:
            copyfile(f"/path/to/vis_config.py", f"{save_path}{label}_{data_set}/configuration.py")
        else:
            copyfile(f"/path/to/vis_config.py", f"{save_path}{label}_{data_set}/configuration.py")

    # set model information files
    mean_file = info_path + "k" + repetition_id + "/mean.npy"
    std_file = info_path + "k" + repetition_id + "/std.npy"
    L = os.listdir(info_path + "k" + repetition_id + "/")
    L.sort()
    model_file = info_path + "k" + repetition_id + "/" + L[-1]

    # load mean + std + model
    mean = np.load(mean_file)
    std = np.load(std_file)
    model = load_model(model_file, compile=False)

    if non_image_features == True:
        non_image_mean_file = info_path + "k" + repetition_id + "/non_image_mean.npy"
        non_image_std_file = info_path + "k" + repetition_id + "/non_image_std.npy"
        non_image_mean = np.load(non_image_mean_file)
        non_image_std = np.load(non_image_std_file)
    else:
        non_image_mean = 0
        non_image_std = 0

    # select subjects for this run
    subjects = all_subjects[run]

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

    # get gradcam visualizations
    run_visualization(subjects, data_path, run_path, mean, std, model, class0, class1, class2, run, all_features, non_image_mean, non_image_std)

    print('\nend')



def run_visualization(subjects, data_path, run_path, mean, std, model, class0, class1, class2, run, all_features, non_image_mean, non_image_std):
    """
    Performs a gradcam analysis of multiple subjects of a specified class.

        INPUT:
            subjects - the subject IDs of the subjects to be processed in this run
            data_path - the path in which the data of the subjects can be found
            run_path - the path in which all output of this run should be saved
            mean - the voxel-wise mean used for normalizing the images
            std - the voxel-wise std used for normalizing the images
            model - the model for which the gradcam should be computed
            class0 - which class is 0
            run - to which run this gradcam analysis belongs

        OUTPUT:
            saves the mean and variation of the Grad-CAM map averaged over the subjects in this run

    """

    # get image shape
    if non_image_features == False:
        X, Y, Z = model.input_shape[1], model.input_shape[2], model.input_shape[3]
    else:
        X, Y, Z = model.input_shape[0][1], model.input_shape[0][2], model.input_shape[0][3]
        feature_list = []
        feature_list.append("Time_feature")
        feature_list.append("Current_diagnosis")

    # get name of layer to visualize
    gc_layer_name, input_layer_1, input_layer_2, dense_layer = get_layer(model, gc_layer)

    # write the subjects of this run in a txt file
    file = open(f"{run_path}subjects_run{run}.txt", 'w')
    file.write(f"\nRUN {run} - Analysis of {class_limit} {label} subjects\n")
    print(f"\nRUN {run} - Analysis of {class_limit} {classification_type} classified {label} subjects\n")

    # allocate batch of images
    images = np.zeros((len(subjects), X, Y, Z))


    if non_image_features == True:
        N = model.input_shape[1][1]#, model.input_shape[0][2], model.input_shape[0][3]
        non_images = np.zeros((len(subjects), N,))
        #X_non_image = np.empty((self.batch_size, *self.dim_non_image,))

    # loop over all subjects in this run
    for i, subject in enumerate(subjects):
        this_row = pd.DataFrame(all_features[all_features["Link"] == subject])
        idx = this_row.index.item()
        ptid_viscode = this_row.loc[idx, "PTID_VISCODE"]

        # get image path
        img_file = f"{data_path}{ptid_viscode}.h5py"
        file.write(f"{i} - subject {subject}\n")
        print(f"\n{i} - Working on subject: {subject}")

        # load + normalize image and store in batch
        image = load_image(img_file, mean, std, ptid_viscode)
        images[i] = image

        if non_image_features == True:
            non_image_columns = this_row[this_row.columns.intersection(feature_list)].to_numpy()
            non_images[i,] = non_image_columns[0]
            non_images[i,] = np.subtract(non_images[i,], non_image_mean)
            non_images[i,] = np.divide(non_images[i,], non_image_std)

            #X = [np.array(exp_im), np.array(X_non_image)]

    # create array indicating the class of each images
    if task == "AD":
        if label == class0:
            cls = 0
        elif label == class1:
            cls = 1
    elif task == "MCI":
        if label == class0:
            cls = 0
        elif label == class1:
            cls = 1
    else:
        if label == class0:
            cls = 0
        elif label == class1:
            cls = 1
        elif label == class2:
            cls = 2
    classes = [cls] * len(subjects)

    # expand image batch to network input shape
    images = np.expand_dims(images, axis=5)

    # compute gradcam mean and variation of the subjects in this run
    gradcam_mean, gradcam_var = grad_cam_batch(model, images, classes, gc_layer_name, X, Y, Z, all_features, non_image_mean, non_image_std, non_images, input_layer_1, input_layer_2, dense_layer)

    # if last run contains less subjects than class limit these should weigh less in the total average
    if len(subjects) is not class_limit:
        weight_factor = len(subjects) / class_limit
        gradcam_mean = gradcam_mean * weight_factor
        gradcam_var = gradcam_var * weight_factor

    # save mean + var gradcams
    np.save(f"{run_path}gradcam_c{gc_layer}_{label}.npy", gradcam_mean)
    np.save(f"{run_path}gradcam-VAR_c{gc_layer}_{label}.npy", gradcam_var)

    file.close()


def get_layer(model, vis_layer):
    """
    Returns the name of the conv3D layer to visualize
    which should be specified in the config file with an integer
    """
    conv_cnt = 0
    input_cnt = 0
    dense_cnt = 0
    input_layer_2 = "na"
    for layer in model.layers:
        if layer.name[:5] == 'input' and input_cnt == 1:
            input_layer_2 = layer.name
        if layer.name[:5] == 'input' and input_cnt == 0:
            input_layer_1 = layer.name
            input_cnt += 1
        if layer.name[:5] == 'dense' and dense_cnt == 0:
            dense_layer = layer.name
            dense_cnt += 1
        if layer.name[:6] == 'conv3d':
            conv_cnt += 1
            if conv_cnt == vis_layer:
                layer_name = layer.name
    return layer_name, input_layer_1, input_layer_2, dense_layer



def load_image(img_file, mean, std, subject):
    """
    Load and normalize image with input mean and std.
    """
    with h5py.File(img_file, 'r') as hf:
        x = hf[subject][:]
    x = np.subtract(x, mean)
    x = np.divide(x, (std + 1e-10))

    return x



def grad_cam_batch(model, images, classes, layer_name, X, Y, Z, all_features, non_image_mean, non_image_std, non_images, input_layer_1, input_layer_2, dense_layer):
    """
    GradCAM method to process multiple images in one run.

        INPUT:
            model - the model for which the gradcams should be computed
            images - the batch of images for which the gradcams should be computed
            classes - an array indicating the classes corresponding to the image batch
            X, Y, Z - the input shapes of the images

        OUTPUT
            cam_mean - the average gradcam of the image batch
            cam_var - the variation of the mean gradcam of the image batch
    """

    # get loss, output and gradients
    loss = tf.gather_nd(model.output, np.dstack([range(images.shape[0]), classes])[0])
    first_input = model.get_layer(input_layer_1).input
    second_input = model.get_layer(input_layer_2).input
    layer_output = model.get_layer(layer_name).output
    dense_layer_output = model.get_layer(dense_layer).output
    grads = K.gradients(loss, layer_output)[0]
    dense_grads = K.gradients(loss, dense_layer_output)[0]

    # calculate class activation maps for image batch
    if non_image_features == False:
        # create gradient function
        gradient_fn = K.function([first_input, K.learning_phase()], [layer_output, grads])
        conv_output, grads_val = gradient_fn([images, learning_phase])
    else:
        # create gradient function
        gradient_fn = K.function([first_input, second_input, K.learning_phase()], [layer_output, grads])
        conv_output, grads_val = gradient_fn([images, non_images, learning_phase])
        # create gradient function for dense layer
        dense_gradient_fn = K.function([first_input, second_input, K.learning_phase()], [dense_layer_output, dense_grads])
        dense_conv_output, dense_grads_val = dense_gradient_fn([images, non_images, learning_phase])

    weights = np.mean(grads_val, axis=(1, 2, 3))
    cams = np.einsum('ijklm,im->ijkl', conv_output, weights)
    #dense_weights = np.mean(dense_grads_val, axis=(1))
    dense_cams = np.einsum('ij,ij->i', dense_conv_output, dense_grads_val)

    # process CAMs
    new_cams = np.empty((images.shape[0], X, Y, Z))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i) + 1e-10)
        new_cams[i] = resize(cam_i, (X, Y, Z), order=1, mode='constant', cval=0, anti_aliasing=False)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    # process CAMs for dense layer
    new_cams_dense = np.empty((images.shape[0], X, Y, Z))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i) + 1e-10)
        new_cams[i] = resize(cam_i, (X, Y, Z), order=1, mode='constant', cval=0, anti_aliasing=False)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    # calculate mean and variation
    cam_mean = np.mean(new_cams, axis=0)
    cam_var = np.var(new_cams, axis=0)

    return cam_mean, cam_var


def create_data_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()
