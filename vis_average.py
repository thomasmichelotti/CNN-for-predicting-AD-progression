import sys
import numpy as np
import nibabel as nib
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.transforms as tsf
from skimage.transform import resize

from vis_config import class_limit, gc_layer, roi, data_set, label, classification_type, server, model_id, repetition_id, mask_factor, interval


def main():
    """
    Calculates the average Grad-CAM image of multiple runs.
    Converts this final Grad-CAM image to nifti format mapped to MNI152.
    """

    # set info paths
    if server:
        misclassification_path = f"/path/to/D_OutputFiles/F_GradCAM_output/A_Misclassifications/{model_id}_k{repetition_id}/"
        main_path = f"/path/to/D_OutputFiles/F_GradCAM_output/B_Main/{model_id}_k{repetition_id}/"
        save_path = f"/path/to/D_OutputFiles/F_GradCAM_output/C_Average/{model_id}_k{repetition_id}/{label}_{data_set}/"
        template_file = f"/path/to/A_DataFiles/GradCAM/downsampled_brain_mask_in_template_space.h5py"
        mni_file = f"/path/to/A_DataFiles/GradCAM/MNI152_T1_1mm_brain.nii.gz"
    else:
        misclassification_path = f"/path/to/D_OutputFiles/F_GradCAM_output/A_Misclassifications/{model_id}_k{repetition_id}/"
        main_path = f"/path/to/D_OutputFiles/F_GradCAM_output/B_Main/{model_id}_k{repetition_id}/"
        save_path = f"/path/to/D_OutputFiles/F_GradCAM_output/C_Average/{model_id}_k{repetition_id}/{label}_{data_set}/"
        template_file = f"/path/to/A_DataFiles/GradCAM/downsampled_brain_mask_in_template_space.h5py"
        mni_file = f"/path/to/A_DataFiles/GradCAM/MNI152_T1_1mm_brain.nii.gz"

    create_data_directory(save_path)
    # load subject file
    all_subjects = np.load(f"{misclassification_path}{classification_type}_classified_subjects_{label}_{data_set}.npy", allow_pickle=True)
    k_splits = all_subjects.shape[0]
    #k_splits = 1                      #for testing with fewer runs
    print(k_splits)

    gc_mean = 0
    gc_var = 0
    #gbs = 0

    # loop over all subject splits
    for k in range(0, k_splits):

        if server:
            # load the gradcam mean file of each split and add this to the gc_mean variable
            gc_img_mean = np.load(f"{main_path}{label}_{data_set}/run{k}_{roi}_n{class_limit}_gc{gc_layer}_{label}/gradcam_c{gc_layer}_{label}.npy")
            # load the gradcam var file of each split and add this to the gc_var variable
            gc_img_var = np.load(f"{main_path}{label}_{data_set}/run{k}_{roi}_n{class_limit}_gc{gc_layer}_{label}/gradcam-VAR_c{gc_layer}_{label}.npy")

        else:
            # load the gradcam mean file of each split and add this to the gc_mean variable
            gc_img_mean = np.load(f"{main_path}{label}_{data_set}/run{k}_{roi}_n{class_limit}_gc{gc_layer}_{label}/gradcam_c{gc_layer}_{label}.npy")
            # load the gradcam var file of each split and add this to the gc_var variable
            gc_img_var = np.load(f"{main_path}{label}_{data_set}/run{k}_{roi}_n{class_limit}_gc{gc_layer}_{label}/gradcam-VAR_c{gc_layer}_{label}.npy")

        gc_mean += gc_img_mean
        gc_var += gc_img_var

    # map the mean gradcam to MNI152 format and store nifti file
    nii_gc_mean = to_mni152(gc_mean / k_splits, template_file, mni_file)
    nib.save(nii_gc_mean, f"{save_path}MEAN_{roi}_n{k_splits * class_limit}_{data_set}_gradcam_c{gc_layer}_{label}.nii")

    # map the var gradcam to MNI152 format and store nifti file
    nii_gc_var = to_mni152(gc_var / k_splits, template_file, mni_file)
    nib.save(nii_gc_var, f"{save_path}VAR_{roi}_n{k_splits * class_limit}_{data_set}_gradcam_c{gc_layer}_{label}.nii")


    # Create images
    custom_cmap_grey = clr.LinearSegmentedColormap.from_list('custom greys', [(1, 1, 1), (.7, .7, .7)], N=256)

    image = gc_mean / k_splits
    # test_side = plt.imshow(image[18, :, :], cmap="YlOrRd")
    # plt.axis('off')
    # plt.show()
    # test_front = plt.imshow(image[:, 23, :], cmap="YlOrRd")
    # plt.axis('off')
    # plt.show()
    # test_top = plt.imshow(image[:, :, 19], cmap="YlOrRd")
    # plt.axis('off')
    # plt.show()

    image_max = np.max(image)
    image_min = np.min(image)
    image_mean = np.mean(image)

    def smallestN_indices_argparitition(a, N, maintain_order=False):
        idx = np.argpartition(a.ravel(), N)[:N]
        if maintain_order:
            idx = idx[a.ravel()[idx].argsort()]
        return np.stack(np.unravel_index(idx, a.shape)).T

    number_of_datapoints = image.shape[0] * image.shape[1] * image.shape[2]
    idx = smallestN_indices_argparitition(image, int(np.round(number_of_datapoints * (1-mask_factor))), maintain_order=False)
    #image[np.where(image < mask_factor*image_max)] = 0
    for i in range(idx.shape[0]):
        image[tuple(idx[i,:])] = 0

    if server:
        template = nib.load(f"/path/to/A_DataFiles/GradCAM/brain_mask_in_template_space.nii.gz").get_fdata()
    else:
        template = nib.load(f"/path/to/A_DataFiles/GradCAM/brain_mask_in_template_space.nii.gz").get_fdata()
    mni_template = nib.load(mni_file).get_fdata()
    mask = np.zeros(template.shape)
    mask[np.where(template != 0)] = 1
    l = np.where(mask != 0)
    # determine the boundaries and corresponding dimensions
    minimum = (min(l[0]), min(l[1]), min(l[2]))
    maximum = (max(l[0]), max(l[1]), max(l[2]))
    filtered_template = mni_template[minimum[0]:(maximum[0] + 1), minimum[1]:(maximum[1] + 1), minimum[2]:(maximum[2] + 1)]

    upsampled_image = resize(image, filtered_template.shape)
    #upsampled_image = resize(image, mni_template.shape)
    #upsampled_image = resize(image, (143, 179, 148))
    #upsampled_image = resize(image, (182, 218, 182))
    masked_upsampled_image = np.ma.masked_where(upsampled_image == 0, upsampled_image)

    old_reference_point = np.asarray([92, 110, 92])
    new_reference_point = old_reference_point - np.asarray(minimum) - np.asarray([1,1,1])

    #base_side = plt.gca().transData
    #rot = tsf.Affine2D().rotate_deg(90)
    image_template_side = plt.imshow(filtered_template[new_reference_point[0],:,:], cmap=custom_cmap_grey)
    # plt.show()
    image_side = plt.imshow(masked_upsampled_image[new_reference_point[0], :, :], cmap="YlOrRd")#, alpha=alpha[new_reference_point[0], :, :]) #YlOrRd, Reds, OrRd
    plt.axis('off')
    plt.show()
    image_side.figure.savefig(save_path + f"{interval}_{label}_image_side.png")
    plt.close()

    image_template_front = plt.imshow(filtered_template[:, new_reference_point[1], :], cmap=custom_cmap_grey)
    # plt.show()
    image_front = plt.imshow(masked_upsampled_image[:, new_reference_point[1], :], cmap="YlOrRd")#, alpha=alpha[:, new_reference_point[1], :])
    plt.axis('off')
    plt.show()
    image_front.figure.savefig(save_path + f"{interval}_{label}_image_front.png")
    plt.close()

    image_template_top = plt.imshow(filtered_template[:, :, new_reference_point[2]], cmap=custom_cmap_grey)
    # plt.show()
    image_top = plt.imshow(masked_upsampled_image[:, :, new_reference_point[2]], cmap="YlOrRd")#, alpha=alpha[:, :, new_reference_point[2]])
    plt.axis('off')
    plt.show()
    image_top.figure.savefig(save_path + f"{interval}_{label}_image_top.png")
    plt.close()




def to_mni152(image, template_file, mni_file):
    """
    This function converts a given numpy matrix to a nifti file mapped to MNI152.

        INPUT:
            image - gets a 3D numpy matrix as input image

        OUTPUT:
            nii_im - returns a 3D nifti file with MNI152 mapping
    """

    # create mask of brain
    #template = nib.load(template_file).get_fdata()
    with h5py.File(template_file, 'r') as hf:
        template = hf["downsampled_brain_mask_in_template_space"][:]
    #X_image[i,] = np.reshape(x, (self.dim[0], self.dim[1], self.dim[2], self.n_channels))
    mask = np.zeros(template.shape)
    mask[np.where(template != 0)] = 1

    # get only real data points
    l = np.where(mask != 0)

    # determine the boundaries and corresponding dimensions
    minimum = (min(l[0]), min(l[1]), min(l[2]))
    maximum = (max(l[0]), max(l[1]), max(l[2]))

    # extract data points corresponding to mask
    imagePad = np.zeros(mask.shape)
    imagePad[minimum[0]:(maximum[0] + 1), minimum[1]:(maximum[1] + 1), minimum[2]:(maximum[2] + 1)] = image

    # scale all values between 0 and 1
    imagePad = imagePad / imagePad.max()

    # map to mni152
    mni = nib.load(mni_file)
    nii_im = nib.Nifti1Image(imagePad, affine=mni.affine)

    return nii_im


def create_data_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    main()