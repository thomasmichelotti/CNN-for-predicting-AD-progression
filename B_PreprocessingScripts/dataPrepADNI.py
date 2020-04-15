# Prepare data script for longitudinal predictions.
import os
import pandas as pd
import numpy as np
from datetime import date, datetime
#from dateutil.relativedelta import relativedelta
from calendar import isleap
from itertools import islice
from matplotlib import pyplot

# Settings
seed = 1

# Specify input paths f"path/to/"
D1_D2_path = f"path/to/D2_D3_file"
D3_path = f"path/to/D3_file"
mri_avail = f"path/to/MRI_availability.txt"


def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


# Read complete datafile
D1_D2 = pd.read_csv(D1_D2_path)



# For some baseline observations, the DXCHANGE column is not filled even though the diagnosis is available in the DX_bl column. DXCHANGE is filled wherever possible.
no_dxchange = D1_D2[D1_D2["DXCHANGE"].isnull()]
idx_bl_missing_diagnosis = no_dxchange[no_dxchange["VISCODE"] == "bl"]
RID = idx_bl_missing_diagnosis["RID"].copy()
uRIDs = np.unique(RID)
for i, row in idx_bl_missing_diagnosis.iterrows():
    if idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "CN" or idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "SMC":
        D1_D2.loc[i, "DXCHANGE"] = 1
    elif idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "EMCI" or idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "LMCI":
        D1_D2.loc[i, "DXCHANGE"] = 2
    elif idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "AD":
        D1_D2.loc[i, "DXCHANGE"] = 3


# Change Diagnosis column
idx_mci = D1_D2['DXCHANGE'] == 4
D1_D2.loc[idx_mci, 'DXCHANGE'] = 2
idx_ad = D1_D2['DXCHANGE'] == 5
D1_D2.loc[idx_ad, 'DXCHANGE'] = 3
idx_ad = D1_D2['DXCHANGE'] == 6
D1_D2.loc[idx_ad, 'DXCHANGE'] = 3
idx_cn = D1_D2['DXCHANGE'] == 7
D1_D2.loc[idx_cn, 'DXCHANGE'] = 1
idx_mci = D1_D2['DXCHANGE'] == 8
D1_D2.loc[idx_mci, 'DXCHANGE'] = 2
idx_cn = D1_D2['DXCHANGE'] == 9
D1_D2.loc[idx_cn, 'DXCHANGE'] = 1
D1_D2 = D1_D2.rename(columns={'DXCHANGE': 'Diagnosis'})
# Change age
D1_D2['AGE'] += D1_D2['Month_bl'] / 12.
# Sort by RID and Age
D1_D2_sorted = D1_D2.sort_values(["RID", "AGE"], ascending=[True, True])
print("Total size: ", D1_D2_sorted.shape)

# Rename columns with non-imaging features
D1_D2_sorted = D1_D2_sorted.rename(columns={'ABETA_UPENNBIOMK9_04_19_17': 'ABETA'})
D1_D2_sorted = D1_D2_sorted.rename(columns={'TAU_UPENNBIOMK9_04_19_17': 'TAU'})
D1_D2_sorted = D1_D2_sorted.rename(columns={'PTAU_UPENNBIOMK9_04_19_17': 'PTAU'})

# Select columns that will be used
D1_D2_selected_columns = D1_D2_sorted[["RID", "PTID", "VISCODE", "Diagnosis", "AGE", "EXAMDATE", "PTGENDER", "ABETA", "TAU", "PTAU", "MMSE", "ADAS13"]]

# Replace empty strings by NaN
D1_D2_selected_columns = D1_D2_selected_columns.replace(r'^\s*$', np.nan, regex=True)

print("COUNT MISSING:")
print(D1_D2_selected_columns.isnull().sum())

# Merge PTID and VISCODE
D1_D2_selected_columns["PTID_VISCODE"] = D1_D2_selected_columns["PTID"] + "_" + D1_D2_selected_columns["VISCODE"]

# Create MRI availability column
MRI_file_names = []
with open(mri_avail, 'r') as filehandle:
    MRI_file_names = [current_place.rstrip() for current_place in filehandle.readlines()]
print("Total MRI availability: ", len(MRI_file_names))
MRI_file_names = [w.replace(".nii.gz", "") for w in MRI_file_names]
D1_D2_selected_columns["MRI_availability"] = np.where(D1_D2_selected_columns.PTID_VISCODE.isin(MRI_file_names), True, False)
print("Total MRI availability: ", D1_D2_selected_columns["MRI_availability"].value_counts())


# # Impute missing diagnosis when possible (only impute when the last known diagnosis before and after a missing diagnosis are the same: 1-nan-1 -> 1-1-1)
# impute_diagnosis = False
# if impute_diagnosis:
#     D1_D2_imputed = D1_D2_selected_columns.copy()
#     RID = D1_D2_imputed["RID"].copy()
#     uRIDs = np.unique(RID)
#     for i in range(len(uRIDs)):
#         print("i: ", i)
#         idx = RID == uRIDs[i]
#         D_temp = D1_D2_imputed.loc[idx]
#         prev_diagnosis = 0
#         loc_prev_diagnosis = 0
#         nan_count_temp = 0
#         nan_count_total = 0
#         nan_loc = []
#         #for j in range(len(D_temp)):
#         for j, row in D_temp.iterrows():
#             if np.isnan(D_temp.loc[j, "Diagnosis"]):
#                 nan_count_temp += 1
#                 nan_count_total += 1
#                 nan_loc.append(j)
#             elif D_temp.loc[j, "Diagnosis"] == 1.0:
#                 if prev_diagnosis == 1.0:
#                     if nan_loc:
#                         for nan in nan_loc:
#                             print(D_temp["Diagnosis"])
#                             D_temp.loc[nan, "Diagnosis"] = 1.0
#                             print(D_temp["Diagnosis"])
#                         nan_loc = []
#                         nan_count_temp = 0
#                 else:
#                     nan_loc = []
#                 prev_diagnosis = 1.0
#                 loc_prev_diagnosis = j
#             elif D_temp.loc[j, "Diagnosis"] == 2.0:
#                 if prev_diagnosis == 2.0:
#                     if nan_loc:
#                         for nan in nan_loc:
#                             print(D_temp["Diagnosis"])
#                             D_temp.loc[nan, "Diagnosis"] = 2.0
#                             print(D_temp["Diagnosis"])
#                         nan_loc = []
#                         nan_count_temp = 0
#                 else:
#                     nan_loc = []
#                 prev_diagnosis = 2.0
#                 loc_prev_diagnosis = j
#             elif D_temp.loc[j, "Diagnosis"] == 3.0: # Also change 3 - nan - nan to 3 - 3 - 3?
#                 if prev_diagnosis == 3.0:
#                     if nan_loc:
#                         for nan in nan_loc:
#                             print(D_temp["Diagnosis"])
#                             D_temp.loc[nan, "Diagnosis"] = 3.0
#                             print(D_temp["Diagnosis"])
#                         nan_loc = []
#                         nan_count_temp = 0
#                 else:
#                     nan_loc = []
#                 prev_diagnosis = 3.0
#                 loc_prev_diagnosis = j
#         D1_D2_imputed.loc[idx] = D_temp
# else:
#     D1_D2_imputed = D1_D2_selected_columns.copy()




# This function will return the datetime in items which is the closest to the date pivot.
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# Additional column that indicates which matches have imputed diagnosis??

# Impute missing diagnosis everywhere (choose diagnosis of nearest available observation of same subject)
impute_closest_diagnosis = False
if impute_closest_diagnosis:
    D1_D2_imputed = D1_D2_selected_columns.copy()
    RID = D1_D2_imputed["RID"].copy()
    uRIDs = np.unique(RID)
    for i in range(len(uRIDs)):
        print("i: ", i)
        idx = RID == uRIDs[i]
        D_temp = D1_D2_imputed.loc[idx]
        D_temp_nan = D_temp[D_temp["Diagnosis"].isnull()]
        D_temp_not_nan = D_temp[~D_temp["Diagnosis"].isnull()]
        available_dates = D_temp_not_nan["EXAMDATE"].tolist()
        available_dates_formatted = [datetime.strptime(x, "%Y-%m-%d") for x in available_dates]
        for j, row in D_temp_nan.iterrows():
            exam_date = D_temp_nan.loc[j, "EXAMDATE"]
            date_formatted = datetime.strptime(exam_date, "%Y-%m-%d")
            closest_date = nearest(available_dates_formatted, date_formatted)
            k = D_temp_not_nan.index[D_temp_not_nan["EXAMDATE"] == datetime.strftime(closest_date, "%Y-%m-%d")]
            D_temp.loc[j, "Diagnosis"] = D_temp_not_nan.loc[k, "Diagnosis"].values
        D1_D2_imputed.loc[idx] = D_temp
else:
    D1_D2_imputed = D1_D2_selected_columns.copy()


# Do imputation for MMSE
impute_mmse = False
if impute_mmse:
    D1_D2_imputed_mmse = D1_D2_imputed.copy()
    RID = D1_D2_imputed_mmse["RID"].copy()
    uRIDs = np.unique(RID)
    for i in range(len(uRIDs)):
        print("i: ", i)
        idx = RID == uRIDs[i]
        D_temp = D1_D2_imputed_mmse.loc[idx]
        D_temp_nan = D_temp[D_temp["MMSE"].isnull()]
        D_temp_not_nan = D_temp[~D_temp["MMSE"].isnull()]
        available_dates = D_temp_not_nan["EXAMDATE"].tolist()
        available_dates_formatted = [datetime.strptime(x, "%Y-%m-%d") for x in available_dates]
        for j, row in D_temp_nan.iterrows():
            exam_date = D_temp_nan.loc[j, "EXAMDATE"]
            date_formatted = datetime.strptime(exam_date, "%Y-%m-%d")
            closest_date = nearest(available_dates_formatted, date_formatted)
            k = D_temp_not_nan.index[D_temp_not_nan["EXAMDATE"] == datetime.strftime(closest_date, "%Y-%m-%d")]
            D_temp.loc[j, "MMSE"] = D_temp_not_nan.loc[k, "MMSE"].values
        D1_D2_imputed_mmse.loc[idx] = D_temp
else:
    D1_D2_imputed_mmse = D1_D2_imputed.copy()


# Do imputation for ADAS13
impute_adas13 = False
if impute_adas13:
    D1_D2_imputed_mmse_adas13 = D1_D2_imputed_mmse.copy()
    RID = D1_D2_imputed_mmse_adas13["RID"].copy()
    uRIDs = np.unique(RID)
    for i in range(len(uRIDs)):
        print("i: ", i)
        idx = RID == uRIDs[i]
        D_temp = D1_D2_imputed_mmse_adas13.loc[idx]
        check = D_temp["ADAS13"].isnull().all()
        if not D_temp["ADAS13"].isnull().all():
            D_temp_nan = D_temp[D_temp["ADAS13"].isnull()]
            D_temp_not_nan = D_temp[~D_temp["ADAS13"].isnull()]
            available_dates = D_temp_not_nan["EXAMDATE"].tolist()
            available_dates_formatted = [datetime.strptime(x, "%Y-%m-%d") for x in available_dates]
            for j, row in D_temp_nan.iterrows():
                exam_date = D_temp_nan.loc[j, "EXAMDATE"]
                date_formatted = datetime.strptime(exam_date, "%Y-%m-%d")
                closest_date = nearest(available_dates_formatted, date_formatted)
                k = D_temp_not_nan.index[D_temp_not_nan["EXAMDATE"] == datetime.strftime(closest_date, "%Y-%m-%d")]
                D_temp.loc[j, "ADAS13"] = D_temp_not_nan.loc[k, "ADAS13"].values
            D1_D2_imputed_mmse_adas13.loc[idx] = D_temp
else:
    D1_D2_imputed_mmse_adas13 = D1_D2_imputed_mmse.copy()




# Save current diagnosis in new column before matching features at time t to labels at time t+1
D1_D2_imputed_mmse_adas13["Current_diagnosis"] = D1_D2_imputed_mmse_adas13["Diagnosis"]

# Create column that will indicate the PTID, the VISCODE of the MRI scan that is linked, and the VISCODE of the diagnosis that the MRI scan is linked to
D1_D2_imputed_mmse_adas13["Link"] = D1_D2_imputed_mmse_adas13["PTID_VISCODE"]

# Create column that indicates VISCODE of the matched diagnosis
D1_D2_imputed_mmse_adas13["Matched_VISCODE"] = D1_D2_imputed_mmse_adas13["VISCODE"]

# Create column that indicates which rows are original and which rows are newly created
D1_D2_imputed_mmse_adas13["Original"] = True

# Create column that will indicate time between observation and diagnosis
D1_D2_imputed_mmse_adas13["Time_feature"] = D1_D2_imputed_mmse_adas13["AGE"]

# Match features at time t to labels at time t+1, t+2, ... for all t
match_diagnosis = True
if match_diagnosis:
    RID = D1_D2_imputed_mmse_adas13["RID"].copy()
    column_names = D1_D2_imputed_mmse_adas13.columns
    D1_D2_matched = pd.DataFrame(columns=column_names)
    uRIDs = np.unique(RID)
    for i in range(len(uRIDs)):
        print("i: ", i)
        idx = RID == uRIDs[i]
        D_temp = D1_D2_imputed_mmse_adas13.loc[idx]
        for j, row in D_temp.iterrows():
            print("j: ", j)
            D1_D2_matched = D1_D2_matched.append(D_temp.loc[j])
            next_index = D_temp.index.get_loc(j) + 1
            for k, next_row in D_temp.iloc[next_index:,:].iterrows():
                print("k: ", k)
                new_DF_row = D_temp.loc[j:j].copy()
                new_DF_row.loc[j, "Link"] = new_DF_row.loc[j, "Link"] + "_" + D_temp.loc[k, "VISCODE"]
                new_DF_row.loc[j, "Matched_VISCODE"] = D_temp.loc[k, "VISCODE"]
                new_DF_row.loc[j, "Diagnosis"] = D_temp.loc[k, "Diagnosis"]
                new_DF_row.loc[j, "Time_feature"] = D_temp.loc[k, "AGE"] - D_temp.loc[j, "AGE"]
                new_DF_row.loc[j, "Original"] = False
                D1_D2_matched = D1_D2_matched.append(new_DF_row.loc[j])
else:
    D1_D2_matched = D1_D2_imputed_mmse_adas13.copy()


#intermediate_output_dir = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/all_matches.csv"
intermediate_output_dir = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/all_matches_no_imputation.csv"
new_intermediate_output = False
if new_intermediate_output == True:
    D1_D2_matched.to_csv(intermediate_output_dir, index=False)
else:
    D1_D2_matched = pd.read_csv(intermediate_output_dir)




run_this_part = True
if run_this_part == True:
    print("COUNT MISSING:")
    print(D1_D2_matched.isnull().sum())
    # Remove observations for which no MRI scan is available
    remove_no_mri = True
    if remove_no_mri:
        print("Total size before removing obs. where no MRI available", D1_D2_matched.shape)
        D1_D2_MRI_available = D1_D2_matched[D1_D2_matched["MRI_availability"] == True]
        print("Total size after removing obs. where no MRI available", D1_D2_MRI_available.shape)
    else:
        D1_D2_MRI_available = D1_D2_matched.copy()

    print("COUNT MISSING:")
    print(D1_D2_MRI_available.isnull().sum())

    temp_unique_mri = D1_D2_MRI_available.drop_duplicates(subset='PTID_VISCODE', keep="last")
    print("UNIQUE MRI: ", temp_unique_mri.shape)
    temp_unique_rid = D1_D2_MRI_available.drop_duplicates(subset='RID', keep="last")
    print("UNIQUE RID: ", temp_unique_rid.shape)

    remove_original = True
    if remove_original == True:
        print("Total size before removing original obs.: ", D1_D2_MRI_available.shape)
        D1_D2_original_removed = D1_D2_MRI_available[D1_D2_MRI_available["Original"] == False]
        print("Total size after removing original obs.: ", D1_D2_original_removed.shape)
    else:
        D1_D2_original_removed = D1_D2_MRI_available.copy()

    remove_no_diagnosis = True
    if remove_no_diagnosis:
        print("Total size before removing obs. where no diagnosis available", D1_D2_original_removed.shape)
        D1_D2_diagnosis_available = D1_D2_original_removed[pd.notnull(D1_D2_original_removed["Diagnosis"])]
        print("Total size after removing obs. where no diagnosis available", D1_D2_diagnosis_available.shape)
    else:
        D1_D2_diagnosis_available = D1_D2_original_removed.copy()

    remove_no_curr_diag = True
    if remove_no_curr_diag:
        print("Total size before removing obs. where no current diagnosis available", D1_D2_diagnosis_available.shape)
        D1_D2_diagnosis_available = D1_D2_diagnosis_available[pd.notnull(D1_D2_diagnosis_available["Current_diagnosis"])]
        print("Total size after removing obs. where no current diagnosis available", D1_D2_diagnosis_available.shape)
    # else:
    #     D1_D2_diagnosis_available = D1_D2_diagnosis_available.copy()

    add_converter_column = True
    if add_converter_column == True:
        D1_D2_diagnosis_available["Converter"] = np.where((D1_D2_diagnosis_available["Current_diagnosis"] >= D1_D2_diagnosis_available["Diagnosis"]), False, True)


#intermediate_output_dir_2 = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/all_matches_new.csv"
#intermediate_output_dir_2 = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/all_matches_no_imputation_new.csv"
#D4_output_dir = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/D4_matches.csv"
#D4_output_dir = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/D4_matches_no_imputation.csv"
#new_intermediate_output = False
#if new_intermediate_output == True:
#    rollover_and_non_rollover_without_D4_scans.to_csv(intermediate_output_dir_2, index=False)
#    D4_test_set_sorted.to_csv(D4_output_dir, index=False)
#else:
#    rollover_and_non_rollover_without_D4_scans = pd.read_csv(intermediate_output_dir_2)
#    D4_test_set_sorted = pd.read_csv(D4_output_dir)

# print("Test size D4: ", D4_test_set_sorted.shape)
# print("Left for train - val - test (LB4) split: ", rollover_and_non_rollover_without_D4_scans.shape)

# Total statistics
RID = D1_D2_diagnosis_available["RID"].copy()
uRIDs = np.unique(RID)
number_of_subjects = len(uRIDs)
print("Number of uRIDs: ", number_of_subjects)

MRI = D1_D2_diagnosis_available["PTID_VISCODE"].copy()
uMRIs = np.unique(MRI)
number_of_mris = len(uMRIs)
print("Number of uMRIs: ", number_of_mris)

print("Number of uMatches: ", D1_D2_diagnosis_available.shape)

# Average time
print("Average time feature: ", D1_D2_diagnosis_available["Time_feature"].mean())

# Count obs. that change diagnosis
count_change = D1_D2_diagnosis_available[D1_D2_diagnosis_available["Diagnosis"] != D1_D2_diagnosis_available["Current_diagnosis"]]
print("Number of obs. where diagnosis changed: ", count_change.shape)

# Count uRID where diagnosis changed in this interval
compute_urid_with_diagnosis_change = True
count_urid_with_diagnosis_change = 0
if compute_urid_with_diagnosis_change:
    DF_urid_with_diagnosis_change = D1_D2_diagnosis_available.copy()
    RID = DF_urid_with_diagnosis_change["RID"].copy()
    uRIDs = np.unique(RID)
    for i in range(len(uRIDs)):
        #print("i: ", i)
        idx = RID == uRIDs[i]
        D_temp = DF_urid_with_diagnosis_change.loc[idx]
        D_first = D_temp.head(1)
        index_first = D_first.index.item()
        D_last = D_temp.tail(1)
        index_last = D_last.index.item()
        if D_first.loc[index_first, "Current_diagnosis"] != D_last.loc[index_last, "Diagnosis"]:
            count_urid_with_diagnosis_change += 1

print("uRID's where diagnosis changed between first and final observation: ", count_urid_with_diagnosis_change)

# # Check converter statistics
# converters = D1_D2_diagnosis_available[D1_D2_diagnosis_available["Converter"] == True]
# print("Total amount of converters: ", converters.shape)
# unique_converters = converters.drop_duplicates(subset='PTID_VISCODE', keep="last")
# print("Total amount of unique converters: ", unique_converters.shape)
#
# converters_CN = converters[converters["Diagnosis"] == 1]
# print("Total amount of CN converters: ", converters_CN.shape)
# unique_CN_converters = converters_CN.drop_duplicates(subset='PTID_VISCODE', keep="last")
# print("Total amount of unique CN converters: ", unique_CN_converters.shape)
#
# converters_MCI = converters[converters["Diagnosis"] == 2]
# print("Total amount of MCI converters: ", converters_MCI.shape)
# unique_MCI_converters = converters_MCI.drop_duplicates(subset='PTID_VISCODE', keep="last")
# print("Total amount of unique MCI converters: ", unique_MCI_converters.shape)
#
# converters_AD = converters[converters["Diagnosis"] == 3]
# print("Total amount of AD converters: ", converters_AD.shape)
# unique_AD_converters = converters_AD.drop_duplicates(subset='PTID_VISCODE', keep="last")
# print("Total amount of unique AD converters: ", unique_AD_converters.shape)


# create new columns that indicate the current PTID_VISCODE and the PTID_VISCODE of the matched diagnosis in integers
D1_D2_diagnosis_available["scan_month"] = D1_D2_diagnosis_available["VISCODE"]
D1_D2_diagnosis_available["scan_month"] = D1_D2_diagnosis_available["scan_month"].str.replace("m", "")
D1_D2_diagnosis_available.loc[D1_D2_diagnosis_available["scan_month"] == "bl", "scan_month"] = 0
D1_D2_diagnosis_available["scan_month"] = pd.to_numeric(D1_D2_diagnosis_available["scan_month"])

D1_D2_diagnosis_available["diagnosis_month"] = D1_D2_diagnosis_available["Matched_VISCODE"]
D1_D2_diagnosis_available["diagnosis_month"] = D1_D2_diagnosis_available["diagnosis_month"].str.replace("m", "")
D1_D2_diagnosis_available.loc[D1_D2_diagnosis_available["diagnosis_month"] == "bl", "diagnosis_month"] = 0
D1_D2_diagnosis_available["diagnosis_month"] = pd.to_numeric(D1_D2_diagnosis_available["diagnosis_month"])

D1_D2_diagnosis_available["month_difference"] = D1_D2_diagnosis_available["diagnosis_month"] - D1_D2_diagnosis_available["scan_month"]



# Check converter statistics
converters = D1_D2_diagnosis_available[D1_D2_diagnosis_available["Converter"] == True]
print("Total amount of converters: ", converters.shape)
unique_converters = converters.drop_duplicates(subset='RID', keep="last")
print("Total amount of unique converters: ", unique_converters.shape)




def interval(lb, ub, DF, drop_duplicate_mris, output, non_image_output, interval_format):
    print("INTERVAL ", lb, "-", ub)
    if interval_format == "years":
        DF_interval_1_temp = DF[(DF["Time_feature"] >= lb) & (DF["Time_feature"] < ub)]
    elif interval_format == "months":
        DF_interval_1_temp = DF[(DF["month_difference"] > lb) & (DF["month_difference"] <= ub)]

    # Compute min en max time interval
    min_interval = DF_interval_1_temp["Time_feature"].min()
    print("Min time interval in this interval: ", min_interval)
    max_interval = DF_interval_1_temp["Time_feature"].max()
    print("Max time interval in this interval: ", max_interval)

    # Create histogram for this interval
    time_intervals = DF_interval_1_temp["Time_feature"].as_matrix()
    histogram_ouput_dir = "C:/Users/050522/PycharmProjects/Thomas/"
    bins = np.linspace(np.floor(min_interval), np.ceil(max_interval), 100)
    fig = pyplot.hist(time_intervals, bins, alpha=0.5, label='Interval')
    pyplot.legend(loc="upper right")
    pyplot.title(f"Distribution of time interval in the interval {lb} {ub}")
    pyplot.xlabel("Time interval")
    pyplot.ylabel("Frequency")
    pyplot.savefig(histogram_ouput_dir + f"time_interval_{lb}_{ub}_adni.png")
    pyplot.show()
    pyplot.close()

    # Drop duplicate MRI scans
    if drop_duplicate_mris == True:
        DF_interval_1 = DF_interval_1_temp.drop_duplicates(subset='PTID_VISCODE', keep="last")
    else:
        DF_interval_1 = DF_interval_1_temp

    # uRID's
    RID_interval_1 = DF_interval_1["RID"].copy()
    u_RID_interval_1 = np.unique(RID_interval_1)
    number_of_u_RID_interval_1 = len(u_RID_interval_1)
    print("Number of unique RIDs in this interval ", number_of_u_RID_interval_1)

    # uMRI's
    MRI_interval_1 = DF_interval_1["PTID_VISCODE"].copy()
    u_MRI_interval_1 = np.unique(MRI_interval_1)
    number_of_u_MRI_interval_1 = len(u_MRI_interval_1)
    print("Number of unique MRI scans in this interval: ", number_of_u_MRI_interval_1)

    # uMatches
    print("Number of unique matches in this interval: ", DF_interval_1.shape)

    # Count CN, MCI and AD
    distribution_diagnosis_interval_1 = DF_interval_1["Diagnosis"].value_counts(dropna=False)
    print("Diagnosis distribution in this interval: ", distribution_diagnosis_interval_1)

    # Average time
    print("Average time feature in this interval: ", DF_interval_1["Time_feature"].mean())

    # Count obs. that change diagnosis
    count_change_interval_1 = DF_interval_1[DF_interval_1["Diagnosis"] != DF_interval_1["Current_diagnosis"]]
    print("Number of obs. where diagnosis changed in this interval: ", count_change_interval_1.shape)

    # Count obs. that progress in diagnosis
    count_progress_interval_1 = DF_interval_1[DF_interval_1["Diagnosis"] > DF_interval_1["Current_diagnosis"]]
    print("Number of obs. where diagnosis progressed in this interval: ", count_progress_interval_1.shape)

    # Count uRID where diagnosis changed in this interval
    compute_urid_with_diagnosis_change = True
    count_urid_with_diagnosis_change = 0
    if compute_urid_with_diagnosis_change:
        DF_urid_with_diagnosis_change = DF_interval_1.copy()
        RID = DF_urid_with_diagnosis_change["RID"].copy()
        uRIDs = np.unique(RID)
        for i in range(len(uRIDs)):
            #print("i: ", i)
            idx = RID == uRIDs[i]
            #D_temp = DF_urid_with_diagnosis_change.loc[idx].tail(1)
            #temp_count = D_temp[D_temp["Diagnosis"] != D_temp["Current_diagnosis"]].shape[0]
            #count_urid_with_diagnosis_change = count_urid_with_diagnosis_change + temp_count
            D_temp = DF_urid_with_diagnosis_change.loc[idx]
            D_first = D_temp.head(1)
            index_first = D_first.index.item()
            D_last = D_temp.tail(1)
            index_last = D_last.index.item()
            if D_first.loc[index_first, "Current_diagnosis"] != D_last.loc[index_last, "Diagnosis"]:
                count_urid_with_diagnosis_change += 1

    print("uRID's where diagnosis changed between first and final observation in this interval: ", count_urid_with_diagnosis_change)

    # Count uRID where diagnosis progressed in this interval
    compute_urid_with_diagnosis_progression = True
    count_urid_with_diagnosis_progression = 0
    if compute_urid_with_diagnosis_progression:
        DF_urid_with_diagnosis_progression = DF_interval_1.copy()
        RID = DF_urid_with_diagnosis_progression["RID"].copy()
        uRIDs = np.unique(RID)
        for i in range(len(uRIDs)):
            # print("i: ", i)
            idx = RID == uRIDs[i]
            # D_temp = DF_urid_with_diagnosis_change.loc[idx].tail(1)
            # temp_count = D_temp[D_temp["Diagnosis"] != D_temp["Current_diagnosis"]].shape[0]
            # count_urid_with_diagnosis_change = count_urid_with_diagnosis_change + temp_count
            D_temp = DF_urid_with_diagnosis_progression.loc[idx]
            D_first = D_temp.head(1)
            index_first = D_first.index.item()
            D_last = D_temp.tail(1)
            index_last = D_last.index.item()
            if D_first.loc[index_first, "Current_diagnosis"] < D_last.loc[index_last, "Diagnosis"]:
                count_urid_with_diagnosis_progression += 1

    print("uRID's where diagnosis progressed between first and final observation in this interval: ", count_urid_with_diagnosis_progression)



    # Check converter statistics
    converters = DF_interval_1[DF_interval_1["Converter"] == True]
    print("Total amount of converters: ", converters.shape)
    unique_converters = converters.drop_duplicates(subset='RID', keep="last")
    print("Total amount of unique converters: ", unique_converters.shape)

    converters_CN = converters[converters["Diagnosis"] == 1]
    print("Total amount of CN converters: ", converters_CN.shape)
    unique_converters_CN = converters_CN.drop_duplicates(subset='RID', keep="last")
    print("Total amount of unique CN converters: ", unique_converters_CN.shape)

    converters_MCI = converters[converters["Diagnosis"] == 2]
    print("Total amount of MCI converters: ", converters_MCI.shape)
    unique_converters_MCI = converters_MCI.drop_duplicates(subset='RID', keep="last")
    print("Total amount of unique MCI converters: ", unique_converters_MCI.shape)

    converters_AD = converters[converters["Diagnosis"] == 3]
    print("Total amount of AD converters: ", converters_AD.shape)
    unique_converters_AD = converters_AD.drop_duplicates(subset='RID', keep="last")
    print("Total amount of unique AD converters: ", unique_converters_AD.shape)




    # Specify output paths leaderboard and TADPOLE files
    output_path_leaderboard = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/newPatients/interval_{lb}_{ub}/"

    # Create output directories if they do not already exist
    create_data_directory(output_path_leaderboard)

    if output:
        # Create main output files
        # Labels output
        output_labels = DF_interval_1[["RID", "PTID_VISCODE", "Link", "Diagnosis", "Converter"]]
        output_labels.to_csv(os.path.join(output_path_leaderboard, 'labels.csv'), index=False)

    if non_image_output:
        # Create non-image output files
        # Non-image ouput
        features = DF_interval_1[["RID", "PTID_VISCODE", "Link", "MMSE", "ADAS13", "Current_diagnosis", "Time_feature", "Converter"]]
        features.to_csv(os.path.join(output_path_leaderboard, 'all_non_image_features.csv'), index=False)



drop_duplicate_mris = False
output = False
non_image_output = False
interval_format = "years"          # months / years
interval(0, 1.25, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
interval(1.25, 2.25, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
interval(2.25, 3.25, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(3.25, 5.25, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(5.25, np.inf, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(0, np.inf, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)


#interval(0, 12, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(12, 24, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(24, 36, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(36, 60, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)

print("END")








