
import pandas as pd
import numpy as np
import os

results_path = 'results/'

ignore_folders = ['task1_hparams']

resultfolders = [rf for rf in os.listdir(results_path) if rf not in ignore_folders]
for rfolder in resultfolders:
    exppath = os.path.join(results_path, rfolder)
    endname = 'metrics_test.csv'
    csvfiles = [file for file in os.listdir(exppath) if endname in file]
    if len(csvfiles) == 0:
        endname = 'metrics_val.csv'
        csvfiles = [file for file in os.listdir(exppath) if endname in file]
        if len(csvfiles) == 0:
            continue

    dfs = []
    for file in csvfiles:
        df = pd.read_csv(os.path.join(exppath, file))
        dfs.append(df)

    concat_group_classes = pd.concat(dfs).groupby(level=0)
    mean_df = concat_group_classes.mean()
    std_df = concat_group_classes.std()

    mean_std_df = pd.DataFrame()
    mean_std_df['Class'] = dfs[0]['Class']
    for col in mean_df:
        mean_std_df['Mean ' + col] = mean_df[col]

    for col in mean_df:
        mean_std_df['Std ' + col] = std_df[col]
    
    outfile = csvfiles[0].split('_')[0] + '_combined_' + endname
    outfile = os.path.join(exppath, outfile)
    mean_std_df.to_csv(outfile, index=False)