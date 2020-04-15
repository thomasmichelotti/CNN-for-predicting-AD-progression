# Prepare data script for longitudinal predictions.
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from itertools import islice
from matplotlib import pyplot

# Settings
seed = 1

# Specify input paths
Data_path = f"path/to/Parelsnoer.csv"
# D3_path = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/Non-image_TADPOLE_INPUT_data/TADPOLE_D3.csv"
# mri_avail = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/MRI_availability.txt"


def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


# Read complete datafile
D = pd.read_csv(Data_path, sep=';')



# For some baseline observations, the DXCHANGE column is not filled even though the diagnosis is available in the DX_bl column. DXCHANGE is filled wherever possible.
# no_dxchange = D1_D2[D1_D2["DXCHANGE"].isnull()]
# idx_bl_missing_diagnosis = no_dxchange[no_dxchange["VISCODE"] == "bl"]
# RID = idx_bl_missing_diagnosis["RID"].copy()
# uRIDs = np.unique(RID)
# for i, row in idx_bl_missing_diagnosis.iterrows():
#     if idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "CN" or idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "SMC":
#         D1_D2.loc[i, "DXCHANGE"] = 1
#     elif idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "EMCI" or idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "LMCI":
#         D1_D2.loc[i, "DXCHANGE"] = 2
#     elif idx_bl_missing_diagnosis.loc[i, "DX_bl"] == "AD":
#         D1_D2.loc[i, "DXCHANGE"] = 3


# Change Diagnosis column
# idx_mci = D1_D2['DXCHANGE'] == 4
# D1_D2.loc[idx_mci, 'DXCHANGE'] = 2
# idx_ad = D1_D2['DXCHANGE'] == 5
# D1_D2.loc[idx_ad, 'DXCHANGE'] = 3
# idx_ad = D1_D2['DXCHANGE'] == 6
# D1_D2.loc[idx_ad, 'DXCHANGE'] = 3
# idx_cn = D1_D2['DXCHANGE'] == 7
# D1_D2.loc[idx_cn, 'DXCHANGE'] = 1
# idx_mci = D1_D2['DXCHANGE'] == 8
# D1_D2.loc[idx_mci, 'DXCHANGE'] = 2
# idx_cn = D1_D2['DXCHANGE'] == 9
# D1_D2.loc[idx_cn, 'DXCHANGE'] = 1
# D1_D2 = D1_D2.rename(columns={'DXCHANGE': 'Diagnosis'})


# Change age
#D1_D2['AGE'] += D1_D2['Month_bl'] / 12.
# Sort by RID and Age


#D1_D2_sorted = D1_D2.sort_values(["RID", "AGE"], ascending=[True, True])
print("Total size: ", D.shape)

# Rename columns
D = D.rename(columns={'SYNDR_DIAG.1': 'Diag_1'})
D = D.rename(columns={'SYNDR_DIAG.2': 'Diag_2'})
D = D.rename(columns={'SYNDR_DIAG.3': 'Diag_3'})
D = D.rename(columns={'SYNDR_DIAG.4': 'Diag_4'})
D = D.rename(columns={'SYNDR_DIAG.5': 'Diag_5'})
D = D.rename(columns={'SYNDR_DIAG.6': 'Diag_6'})
D = D.rename(columns={'SYNDR_DIAG.7': 'Diag_7'})

D = D.rename(columns={'CONSULT_DATUM.1': 'Diag_date_1'})
D = D.rename(columns={'CONSULT_DATUM.2': 'Diag_date_2'})
D = D.rename(columns={'CONSULT_DATUM.3': 'Diag_date_3'})
D = D.rename(columns={'CONSULT_DATUM.4': 'Diag_date_4'})
D = D.rename(columns={'CONSULT_DATUM.5': 'Diag_date_5'})
D = D.rename(columns={'CONSULT_DATUM.6': 'Diag_date_6'})
D = D.rename(columns={'CONSULT_DATUM.7': 'Diag_date_7'})

D = D.rename(columns={'MRI_ID_XNAT': 'RID'})
D = D.rename(columns={'MRI_Date_XNAT': 'Scan_date'})


# Select columns that will be used
D_selected_columns = D[["RID", "Scan_date", "Diag_1", "Diag_2", "Diag_3", "Diag_4", "Diag_5", "Diag_6", "Diag_7", "Diag_date_1", "Diag_date_2", "Diag_date_3", "Diag_date_4", "Diag_date_5", "Diag_date_6", "Diag_date_7"]]

# Replace empty strings by NaN
D_selected_columns = D_selected_columns.replace(r'^\s*$', np.nan, regex=True)



print("COUNT MISSING:")
print(D_selected_columns.isnull().sum())



# Create separate row for every diagnosis date per ID and check if diagnosis is always 1, 2, or 3
columns_to_rows = False
if columns_to_rows == True:
    #D_columns_to_rows = D_selected_columns.copy()
    D_columns_to_rows = pd.DataFrame(columns=["RID", "Scan_date", "Diagnosis", "Diag_date", "Current_diagnosis", "Original"])
    RID = D_selected_columns["RID"].copy()
    uRIDs = np.unique(RID)
    for i in range(len(uRIDs)):
        print("i: ", i)
        idx = RID == uRIDs[i]
        D_temp = D_selected_columns.loc[idx]
        # D_temp = D_columns_to_rows[D_columns_to_rows["RID"] == uRIDs[i]]
        # old_DF_row = D_temp.loc[0:0].copy()
        index = D_temp.index.item()
        for j in range(7):
            Diag = "Diag_" + str(j+1)
            Diag_date = "Diag_date_" + str(j+1)
            if j == 0:
                D_columns_to_rows = D_columns_to_rows.append(
                    {"RID": D_temp.loc[index, "RID"], "Scan_date": D_temp.loc[index, "Scan_date"],
                     "Diagnosis": D_temp.loc[index, Diag], "Diag_date": D_temp.loc[index, Diag_date],
                     "Current_diagnosis": D_temp.loc[index, "Diag_1"], "Original": True}, ignore_index=True)
            else:
                D_columns_to_rows = D_columns_to_rows.append(
                    {"RID": D_temp.loc[index, "RID"], "Scan_date": D_temp.loc[index, "Scan_date"],
                     "Diagnosis": D_temp.loc[index, Diag], "Diag_date": D_temp.loc[index, Diag_date],
                     "Current_diagnosis": D_temp.loc[index, "Diag_1"], "Original": False}, ignore_index=True)


#intermediate_output_dir = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedData/all_matches.csv"
intermediate_output_dir = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedDataParelsnoer/all_matches_no_imputation.csv"
new_intermediate_output = False
if new_intermediate_output == True:
    D_columns_to_rows.to_csv(intermediate_output_dir, index=False)
else:
    D_columns_to_rows = pd.read_csv(intermediate_output_dir)




run_this_part = True
if run_this_part == True:
    print("COUNT MISSING:")
    print(D_columns_to_rows.isnull().sum())
    # Remove observations for which no MRI scan is available
    remove_no_mri = False # MRI is always available in this dataset
    if remove_no_mri:
        print("Total size before removing obs. where no MRI available", D_columns_to_rows.shape)
        D_MRI_available = D_columns_to_rows[pd.notnull(D_columns_to_rows["MRI_availability"])]
        print("Total size after removing obs. where no MRI available", D_MRI_available.shape)
    else:
        D_MRI_available = D_columns_to_rows.copy()

    print("COUNT MISSING:")
    print(D_MRI_available.isnull().sum())

    temp_unique_rid = D_MRI_available.drop_duplicates(subset='RID', keep="last")
    print("UNIQUE RID: ", temp_unique_rid.shape)

    remove_original = True
    if remove_original == True:
        print("Total size before removing original obs.: ", D_MRI_available.shape)
        D_original_removed = D_MRI_available[D_MRI_available["Original"] == False]
        print("Total size after removing original obs.: ", D_original_removed.shape)
    else:
        D_original_removed = D_MRI_available.copy()

    remove_no_diagnosis = True
    if remove_no_diagnosis:
        print("Total size before removing obs. where no diagnosis available", D_original_removed.shape)
        D_diagnosis_available = D_original_removed[pd.notnull(D_original_removed["Diagnosis"])]
        print("Total size after removing obs. where no diagnosis available", D_diagnosis_available.shape)
    else:
        D_diagnosis_available = D_original_removed.copy()

    remove_no_curr_diag = True
    if remove_no_curr_diag:
        print("Total size before removing obs. where no current diagnosis available", D_diagnosis_available.shape)
        D_diagnosis_available = D_diagnosis_available[pd.notnull(D_diagnosis_available["Current_diagnosis"])]
        print("Total size after removing obs. where no current diagnosis available", D_diagnosis_available.shape)
    # else:
    #     D1_D2_diagnosis_available = D1_D2_diagnosis_available.copy()

    # Replace wrong diagnosis values by NaN
    D_diagnosis_available.loc[(D_diagnosis_available["Diagnosis"] < 1) | (D_diagnosis_available["Diagnosis"] > 3), "Diagnosis"] = np.nan
    D_diagnosis_available.loc[(D_diagnosis_available["Current_diagnosis"] < 1) | (D_diagnosis_available["Current_diagnosis"] > 3), "Current_diagnosis"] = np.nan

    if remove_no_diagnosis:
        print("Total size before removing obs. where no diagnosis available", D_diagnosis_available.shape)
        D_diagnosis_available = D_diagnosis_available[pd.notnull(D_diagnosis_available["Diagnosis"])]
        print("Total size after removing obs. where no diagnosis available", D_diagnosis_available.shape)

    if remove_no_curr_diag:
        print("Total size before removing obs. where no current diagnosis available", D_diagnosis_available.shape)
        D_diagnosis_available = D_diagnosis_available[pd.notnull(D_diagnosis_available["Current_diagnosis"])]
        print("Total size after removing obs. where no current diagnosis available", D_diagnosis_available.shape)



    def toYearFraction(date):
        def sinceEpoch(date):  # returns seconds since epoch
            return time.mktime(date.timetuple())

        s = sinceEpoch

        year = date.year
        startOfThisYear = dt(year=year, month=1, day=1)
        startOfNextYear = dt(year=year + 1, month=1, day=1)

        yearElapsed = s(date) - s(startOfThisYear)
        yearDuration = s(startOfNextYear) - s(startOfThisYear)
        fraction = yearElapsed / yearDuration

        return date.year + fraction

    # Match features at time t to labels at time t+1, t+2, ... for all t
    create_time_feature = True
    if create_time_feature:
        # Create column that will indicate time between observation and diagnosis
        D_diagnosis_available["Time_feature"] = D_diagnosis_available["RID"]
        for index, row in D_diagnosis_available.iterrows():
            print(index)
            first_date = D_diagnosis_available.loc[index, "Scan_date"]
            first_date = dt.strptime(first_date, "%m/%d/%Y")
            fractional_year_1 = toYearFraction(first_date)
            second_date = D_diagnosis_available.loc[index, "Diag_date"]
            second_date = dt.strptime(second_date, "%m/%d/%Y")
            fractional_year_2 = toYearFraction(second_date)
            D_diagnosis_available.loc[index, "Time_feature"] = fractional_year_2 - fractional_year_1

    add_converter_column = True
    if add_converter_column == True:
        D_diagnosis_available["Converter"] = np.where((D_diagnosis_available["Current_diagnosis"] >= D_diagnosis_available["Diagnosis"]), False, True)





# Total statistics
RID = D_diagnosis_available["RID"].copy()
uRIDs = np.unique(RID)
number_of_subjects = len(uRIDs)
print("Number of uRIDs: ", number_of_subjects)

# MRI = D_diagnosis_available["PTID_VISCODE"].copy()
# uMRIs = np.unique(MRI)
# number_of_mris = len(uMRIs)
print("Number of uMRIs: ", number_of_subjects)

print("Number of uMatches: ", D_diagnosis_available.shape)

# Average time
print("Average time feature: ", D_diagnosis_available["Time_feature"].mean())

# Count obs. that change diagnosis
count_change = D_diagnosis_available[D_diagnosis_available["Diagnosis"] != D_diagnosis_available["Current_diagnosis"]]
print("Number of obs. where diagnosis changed: ", count_change.shape)

# Count uRID where diagnosis changed in this interval
compute_urid_with_diagnosis_change = True
count_urid_with_diagnosis_change = 0
if compute_urid_with_diagnosis_change:
    DF_urid_with_diagnosis_change = D_diagnosis_available.copy()
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


# Check converter statistics
converters = D_diagnosis_available[D_diagnosis_available["Diagnosis"] > D_diagnosis_available["Current_diagnosis"]]
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
    pyplot.savefig(histogram_ouput_dir + f"time_interval_{lb}_{ub}_parelsnoer.png")
    pyplot.show()
    pyplot.close()

    # Drop duplicate MRI scans
    if drop_duplicate_mris == True:
        DF_interval_1 = DF_interval_1_temp.drop_duplicates(subset='RID', keep="last")
    else:
        DF_interval_1 = DF_interval_1_temp

    # uRID's
    RID_interval_1 = DF_interval_1["RID"].copy()
    u_RID_interval_1 = np.unique(RID_interval_1)
    number_of_u_RID_interval_1 = len(u_RID_interval_1)
    print("Number of unique RIDs in this interval ", number_of_u_RID_interval_1)

    # uMRI's
    # MRI_interval_1 = DF_interval_1["PTID_VISCODE"].copy()
    # u_MRI_interval_1 = np.unique(MRI_interval_1)
    # number_of_u_MRI_interval_1 = len(u_MRI_interval_1)
    print("Number of unique MRI scans in this interval: ", number_of_u_RID_interval_1)

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

    # Count uRID where diagnosis changed in this interval
    compute_urid_with_diagnosis_change = False # not necessary because is the same as obs. that change diagnosis
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

    print("uRID's where diagnosis changed between first and final observation in this interval: ", count_change_interval_1.shape)




    # Check converter statistics
    converters = DF_interval_1[DF_interval_1["Diagnosis"] > DF_interval_1["Current_diagnosis"]]
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
    output_path_leaderboard = f"C:/Users/050522/PycharmProjects/Thomas/A_DataFiles/PreprocessedDataParelsnoer/newPatients/interval_{lb}_{ub}/"

    # Create output directories if they do not already exist
    create_data_directory(output_path_leaderboard)

    if output:
        # Create main output files
        # Labels output
        output_labels = DF_interval_1[["RID", "Diagnosis", "Converter"]]
        output_labels.to_csv(os.path.join(output_path_leaderboard, 'labels.csv'), index=False)

    if non_image_output:
        # Create non-image output files
        # Non-image ouput
        features = DF_interval_1[["RID", "Current_diagnosis", "Time_feature", "Converter"]]
        features.to_csv(os.path.join(output_path_leaderboard, 'all_non_image_features.csv'), index=False)



drop_duplicate_mris = True
output = True
non_image_output = True
interval_format = "years"          # months / years
interval(0, 1.25, D_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
interval(1.25, 2.25, D_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
interval(2.25, 3.25, D_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
interval(3.25, 5.25, D_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(5.25, np.inf, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(0, np.inf, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)


#interval(0, 12, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(12, 24, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(24, 36, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)
#interval(36, 60, D1_D2_diagnosis_available, drop_duplicate_mris, output, non_image_output, interval_format)

print("END")








