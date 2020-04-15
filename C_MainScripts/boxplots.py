import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
def create_data_directory(path):
    """
    Creates new data path if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)

location = "server"             # local / server
sns.set_style("whitegrid")



def create_boxplots(results_name_model_1, results_name_model_2, results_name_model_3, dataset, lb1, lb2):
    if location == "local":
        results_dir_model_1 = f"/path/to/D_OutputFiles/B_Output/" + dataset + results_name_model_1
        results_dir_model_2 = f"/path/to/D_OutputFiles/B_Output/" + dataset + results_name_model_2
        results_dir_model_3 = f"/path/to/D_OutputFiles/B_Output/" + dataset + results_name_model_3
        output_dir = f"/path/to/D_OutputFiles/G_Boxplots/" + dataset + results_name_model_3
    elif location == "server":
        results_dir_model_1 = f"/path/to/D_OutputFiles/" + dataset + results_name_model_1
        results_dir_model_2 = f"/path/to/D_OutputFiles/" + dataset + results_name_model_2
        results_dir_model_3 = f"/path/to/D_OutputFiles/" + dataset + results_name_model_3
        output_dir = f"/path/to/D_OutputFiles/G_Boxplots/" + dataset + results_name_model_3

    create_data_directory(output_dir)

    # load results
    results_model_1 = np.load(results_dir_model_1 + "results.npy", allow_pickle=True)
    results_model_1 = results_model_1.item()
    results_model_2 = np.load(results_dir_model_2 + "results.npy", allow_pickle=True)
    results_model_2 = results_model_2.item()
    results_model_3 = np.load(results_dir_model_3 + "results.npy", allow_pickle=True)
    results_model_3 = results_model_3.item()

    # get results from the three models
    MAUC_overall_model_1 = results_model_1["test"]["mauc"]
    BCA_overall_model_1 = results_model_1["test"]["bca"]
    ARI_overall_model_1 = results_model_1["test"]["ari"]
    MAUC_nc_model_1 = results_model_1["test"]["mauc_non_converters"]
    BCA_nc_model_1 = results_model_1["test"]["bca_non_converters"]
    ARI_nc_model_1 = results_model_1["test"]["ari_non_converters"]
    BCA_c_model_1 = results_model_1["test"]["bca_converters"]
    ARI_c_model_1 = results_model_1["test"]["ari_converters"]

    MAUC_overall_model_2 = results_model_2["test"]["mauc"]
    BCA_overall_model_2 = results_model_2["test"]["bca"]
    ARI_overall_model_2 = results_model_2["test"]["ari"]
    MAUC_nc_model_2 = results_model_2["test"]["mauc_non_converters"]
    BCA_nc_model_2 = results_model_2["test"]["bca_non_converters"]
    ARI_nc_model_2 = results_model_2["test"]["ari_non_converters"]
    BCA_c_model_2 = results_model_2["test"]["bca_converters"]
    ARI_c_model_2 = results_model_2["test"]["ari_converters"]

    MAUC_overall_model_3 = results_model_3["test"]["mauc"]
    BCA_overall_model_3 = results_model_3["test"]["bca"]
    ARI_overall_model_3 = results_model_3["test"]["ari"]
    MAUC_nc_model_3 = results_model_3["test"]["mauc_non_converters"]
    BCA_nc_model_3 = results_model_3["test"]["bca_non_converters"]
    ARI_nc_model_3 = results_model_3["test"]["ari_non_converters"]
    BCA_c_model_3 = results_model_3["test"]["bca_converters"]
    ARI_c_model_3 = results_model_3["test"]["ari_converters"]


    # GROUPED BOXLOTS
    repetitions = len(MAUC_overall_model_1)
    number_of_models = 3

    # Overall grouped boxplots
    models = np.concatenate([repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"], repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"], repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"]])
    metrics = np.concatenate([repetitions * number_of_models * ["MAUC overall"], repetitions * number_of_models * ["BCA overall"], repetitions * number_of_models * ["ARI overall"]])

    MAUC_overall_grouped = np.concatenate([MAUC_overall_model_1, MAUC_overall_model_2, MAUC_overall_model_3])
    BCA_overall_grouped = np.concatenate([BCA_overall_model_1, BCA_overall_model_2, BCA_overall_model_3])
    ARI_overall_grouped = np.concatenate([ARI_overall_model_1, ARI_overall_model_2, ARI_overall_model_3])
    scores_overall_grouped = np.concatenate([MAUC_overall_grouped, BCA_overall_grouped, ARI_overall_grouped])

    df_overall_grouped = pd.DataFrame(np.column_stack([scores_overall_grouped, models, metrics]))
    df_overall_grouped.columns = ['Scores', 'Models', 'Metrics']
    df_overall_grouped = df_overall_grouped.astype({'Scores': 'float64'})

    plt.ylim(lb1, 1)
    plot_overall_grouped = sns.boxplot(data=df_overall_grouped, x="Models", y = "Scores", hue="Metrics", palette="Pastel2", showmeans=True, meanprops={"marker":"D","markerfacecolor":"black", "markeredgecolor":"black"}, showfliers=False)
    plot_overall_grouped.set_ylabel('')
    plot_overall_grouped.set_xlabel('')
    plt.legend(loc='upper left')
    plt.setp(plot_overall_grouped.get_legend().get_texts(), fontsize='8') # for legend text
    plt.setp(plot_overall_grouped.get_legend().get_title(), fontsize='10') # for legend title
    plt.show()
    plot_overall_grouped.figure.savefig(output_dir + "plot_overall_grouped.png")
    plt.close()


    # Converters and non-converters grouped boxplots
    models_nc_c = np.concatenate([repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"], repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"], repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"], repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"], repetitions * ["Model 1"], repetitions * ["Model 2"], repetitions * ["Model 3"]])
    metrics_nc_c = np.concatenate([repetitions * number_of_models * ["MAUC non-converters"], repetitions * number_of_models * ["BCA non-converters"], repetitions * number_of_models * ["BCA converters"], repetitions * number_of_models * ["ARI non-converters"], repetitions * number_of_models * ["ARI converters"]])

    MAUC_nc_grouped = np.concatenate([MAUC_nc_model_1, MAUC_nc_model_2, MAUC_nc_model_3])
    BCA_nc_grouped = np.concatenate([BCA_nc_model_1, BCA_nc_model_2, BCA_nc_model_3])
    BCA_c_grouped = np.concatenate([BCA_c_model_1, BCA_c_model_2, BCA_c_model_3])
    ARI_nc_grouped = np.concatenate([ARI_nc_model_1, ARI_nc_model_2, ARI_nc_model_3])
    ARI_c_grouped = np.concatenate([ARI_c_model_1, ARI_c_model_2, ARI_c_model_3])

    scores_nc_c_grouped = np.concatenate([MAUC_nc_grouped, BCA_nc_grouped, BCA_c_grouped, ARI_nc_grouped, ARI_c_grouped])

    df_nc_c_grouped = pd.DataFrame(np.column_stack([scores_nc_c_grouped, models_nc_c, metrics_nc_c]))
    df_nc_c_grouped.columns = ['Scores', 'Models', 'Metrics']
    df_nc_c_grouped = df_nc_c_grouped.astype({'Scores': 'float64'})

    plt.ylim(lb2, 1)
    plot_nc_c_grouped = sns.boxplot(data=df_nc_c_grouped, x="Models", y = "Scores", hue="Metrics", palette="Pastel2", showmeans=True, meanprops={"marker":"D","markerfacecolor":"black", "markeredgecolor":"black"}, showfliers=False)
    plot_nc_c_grouped.set_ylabel('')
    plot_nc_c_grouped.set_xlabel('')
    plt.legend(loc='upper left')
    plt.setp(plot_nc_c_grouped.get_legend().get_texts(), fontsize='8') # for legend text
    plt.setp(plot_nc_c_grouped.get_legend().get_title(), fontsize='10') # for legend title
    plt.show()
    plot_nc_c_grouped.figure.savefig(output_dir + "plot_nc_c_grouped.png")
    plt.close()




# specify input and output directories 0-1 year interval
results_name_model_1 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 1
results_name_model_2 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 2
results_name_model_3 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 3
#ADNI
create_boxplots(results_name_model_1, results_name_model_2, results_name_model_3, "B_Output/", -0.5, -0.5)
# PARELSNOER
create_boxplots(results_name_model_1, results_name_model_2, results_name_model_3, "E_PSI_output/", -0.5, -0.5)

# specify input and output directories 1-2 year interval
results_name_model_1 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 1
results_name_model_2 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 2
results_name_model_3 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 3
#ADNI
create_boxplots(results_name_model_1, results_name_model_2, results_name_model_3, "B_Output/", -0.5, -0.5)
# PARELSNOER
create_boxplots(results_name_model_1, results_name_model_2, results_name_model_3, "E_PSI_output/", -0.5, -0.5)

# specify input and output directories 2-3 year interval
results_name_model_1 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 1
results_name_model_2 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 2
results_name_model_3 = ".../"               # CHANGE THIS ONE TO OUTPUT NAME MODEL 3
#ADNI
create_boxplots(results_name_model_1, results_name_model_2, results_name_model_3, "B_Output/", -0.5, -0.5)
# PARELSNOER
create_boxplots(results_name_model_1, results_name_model_2, results_name_model_3, "E_PSI_output/", -0.5, -0.5)




