#######################################

CNN-for-predicting-AD-progression

#######################################

Thesis project MSc. Econometrics and Management Science, specialisation Business Analytics and Quantitative Marketing
Erasmus School of Economics, Erasmus University Rotterdam

Combined with a graduate internship at the Biomedical Imaging Group Rotterdam (BIGR), Erasmus MC


Title: Convolutional Neural Networks for Multiclass Predictions of Alzheimer's Disease Progression

Author: Thomas Michelotti

#######################################

A_DataFiles: 

- This directory is used to store all datafiles before and after running the preprocessing scripts.
Data used in the preparation of this article is obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD. The primary goal of ADNI has been to test whether serial magnetic resonance imaging (MRI), positron emission tomography (PET), other biological markers, and clinical and neuropsychological assessment can be combined to measure the progression of mild cognitive impairment (MCI) and early Alzheimer’s disease (AD).

#######################################

B_Preprocessing_Scripts:

- nii_to_h5py.py is used to convert all images to .h5py format
- get_mri_availability.py is used to create a list of all data points of all individuals for which an MRI scan is available
- dataPrepADNI.py is used to preprocess the ADNI data
- dataPrepParelsnoer.py is used to preprocess the Parelsnoer data

#######################################

C_MainScripts:

- augmentation.py can be used to create a larger training set based on existing images, which is not used in this thesis
- boxplots.py is used to create the boxplots of the prediction performance
- callbacks.py contains the callbacks used for training the CNN
- compute_CI.py can be used to create a confidence interval of the results, which is not used in this thesis
- create_sets.py is used to create training, validation, and test sets
- EpochPerformance.py is used to evaluate prediction performanc after every training epoch
- evalOneSubmissionExtended.py contains a function to calculate BCA
- generator.py contains the data generators for the CNN
- MAUC.py is used to calculate MAUC
- model_allCNN.py is used to create the CNN
- model_selection.py is used to either create a new CNN or load a pre-trained network
- ordinal_categorical_crossentropy.py contains the custom loss function
- plotting.py is used to create several plots related to model performance
- PSI_generators.py contains the Parelsnoer data generators
- PSI_savings.py is used to save Parelsnoer results
- savings.py is used to save ADNI results
- standardise.py is used to standardise the data

#######################################

D_OutputFiles: 

- This directory is used to store all results

#######################################

Visualisation scripts:

- vis_config.py is used to set all parameters for the Grad-CAM visualisation tool
- vis_get_misclassifications.py is used to retrieve all correctly and incorrectly predicted observations
- vis_main.py is used to obtain the Grad-CAM results per batch of 10 images
- vis_average.py is used to average the Grad-CAM results over all batches

#######################################

Configuration files and main files:

- config.py is used to set all parameters for the ADNI data
- main.py should be run to train, validate, and test the CNN on the ADNI data
- config_PSI.py is used to set all parameters for the Parelsnoer data
- main.py should be run to test the CNN on the Parelsnoer data

#######################################

Thesis:

Final version of my Master's thesis

#######################################
