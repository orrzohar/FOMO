import argparse
import os 
import pandas as pd 
import numpy as np

def main(args):
    task_1      = args.Task_1
    task_2      = args.Task_2
    datasets    = args.Datasets
    experiments = args.Experiments
    shots       = args.Shots
    extension   = args.extension



 
    #result_df = pd.DataFrame(columns=['U', 'K', 'PK', 'CK'])
    ## Create name ###
    names  = []
    for dataset in datasets:
        names.extend([f'U_{dataset}', f'K_{dataset}', f'PK_{dataset}', f'CK_{dataset}'])
    names.extend([f'U_mean', f'K_mean', f'PK_mean', f'CK_mean'])

    df   = pd.DataFrame(columns=names )
    df_2 = pd.DataFrame(columns=[f'U_mean', f'K_mean', f'PK_mean', f'CK_mean'])

    for shot in shots:
        for experiment in experiments:
            row     = []
            K_list  = []
            U_list  = []
            PK_list = []
            CK_list = []
            for dataset in datasets:
                file_path = f"{dataset}-{shot}-{experiment}{extension}"

                file_path_task_1  = os.path.join(task_1["file_path"],file_path )
                task_1_df         = pd.read_csv(file_path_task_1)
                try:
                    file_path_task_2  = os.path.join(task_2["file_path"],file_path )
                    task_2_df         = pd.read_csv(file_path_task_2)
                except:
                    print("ERROR")
                    task_2_df = pd.DataFrame({"PK_AP50": [0], "CK_AP50": [0]})

                #for  keys,values in task_1["values"].items():
                K = task_1_df[task_1["values"]["K"]].values[0]
                U = task_1_df[task_1["values"]["U"]].values[0]
                PK = task_2_df[task_2["values"]["PK"]].values[0]
                CK = task_2_df[task_2["values"]["CK"]].values[0]
            

                row.append(U)
                U_list.append(U)
                row.append(K)
                K_list.append(K)
                row.append(PK)
                PK_list.append(PK)
                row.append(CK)
                CK_list.append(CK)

            K_list  = np.array(K_list)
            U_list  = np.array(U_list )
            PK_list = np.array(PK_list)
            CK_list = np.array(CK_list )
            row_2 = [np.mean(U_list),np.mean(K_list ),np.mean(PK_list),np.mean(CK_list )]
            row.extend(row_2)

            df.loc[len(df.index)]    = row
            df_2.loc[len(df.index)]  = row_2
            
    df.index = experiments 
    df       = df.reset_index() 

    df_2.index = experiments 
    df_2       = df_2.reset_index() 
    print(df)

    latex_table = df.to_latex(index=False, float_format='%.1f')

    with open('abilations.tex', 'w') as file:
        file.write(latex_table)

    latex_table = df_2.to_latex(index=False, float_format='%.1f')

    with open('abilations_summary.tex', 'w') as file:
        file.write(latex_table)
    
    
                

            
                

                



if __name__ == "__main__":
    default_task_1       = {"values":{"U":"U_AP50","K":"K_AP50"},      "file_path":os.path.join("..","tmp","ablation","google","owlvit-large-patch14")}
    default_task_2       = {"values":{"PK":"PK_AP50" ,"CK":"CK_AP50"}, "file_path":os.path.join("..","tmp","ablation_2","google","owlvit-large-patch14")}
    default_datasets     = ["AQUA", "DIOR_FIN", "SYNTH", "XRAY", "NEUROSURGICAL_TOOLS_FIN"]
    #default_experiments  = ["no_adapt", "no_att_selection", "no_refinement"]
    default_experiments  = ["no_refinement","no_refinement-no_att_selection","no_refinement-no_att_selection-no_adapt"]
    default_shots        = ["100"]
    default_extension    = ".csv"
    # Aquatic Aerial Game Medical Surgery
    parser = argparse.ArgumentParser(description="Process parameters with default values")
    parser.add_argument("--Task_1",      type=dict, default=default_task_1,      help="Task 1 parameters")
    parser.add_argument("--Task_2",      type=dict, default=default_task_2,      help="Task 2 parameters")
    parser.add_argument("--Datasets",    type=list, default=default_datasets,    help="List of datasets")
    parser.add_argument("--Experiments", type=list, default=default_experiments, help="List of experiments")
    parser.add_argument("--Shots",       type=list, default=default_shots,       help="List of shots")
    parser.add_argument("--extension",   type=str,  default=default_extension,   help="File extension")

    args = parser.parse_args()
    main(args)
