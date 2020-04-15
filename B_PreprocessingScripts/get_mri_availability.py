############################################################
# Run these lines in MobaXterm when connected to Cartesius #
############################################################

# All files
# dir = "/projects/0/emc16367/ADNI/T1w/nuc"
# output_dir = "/home/michelot/listfile.txt"
#
# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
#
# with open(output_dir, 'w') as filehandle:
#     for listitem in onlyfiles:
#         filehandle.write('%s\n' % listitem)

############################################################
# Run these lines in MobaXterm when connected to Cartesius #
############################################################

# Already processed files
# dir = "/projects/0/emc16367/ADNI/Template_space"
# output_dir = "/home/michelot/listfile_processed.txt"
#
# import os
# directories = os.listdir(dir)
#
# with open(output_dir, 'w') as filehandle:
#     for listitem in directories:
#         filehandle.write('%s\n' % listitem)

####################################################################
# Run these lines to compare MRI availability with processed MRI's #
####################################################################

mri_avail = f"path/to/MRI_availability.txt"
mri_processed = f"path/to/listfile_processed.txt"

with open(mri_avail, 'r') as filehandle:
    MRI_file_names = [current_place.rstrip() for current_place in filehandle.readlines()]
MRI_file_names = [w.replace(".nii.gz", "") for w in MRI_file_names]
MRI_file_names.sort()

with open(mri_processed, 'r') as filehandle:
    MRI_processed = [current_place.rstrip() for current_place in filehandle.readlines()]
MRI_processed.sort()

#diff = list(set(MRI_file_names) - set(MRI_processed))
diff = list(set(MRI_processed) - set(MRI_file_names))
diff.sort()
print(diff)

def by_size(words, size):
    return [word for word in words if len(word) == size]

over100months = by_size(MRI_file_names, 15)
over100months.sort()

print(over100months)








