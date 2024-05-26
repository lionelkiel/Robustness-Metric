import numpy as np
import pandas as pd
import os

path = os.path.abspath("C:\\Users\\lkiel\\PycharmProjects\\RobustnessMetric\\Datasets\\threshold_distributions_FOMLAS")

if os.path.exists(os.path.join(path, "df_epsilon.csv")) and os.path.exists(os.path.join(path, "df_epsilon_crit.csv")):
    df_epsilon = pd.read_csv(os.path.join(path, "df_epsilon.csv"))
    df_epsilon_crit = pd.read_csv(os.path.join(path, "df_epsilon_crit.csv"))
else:
    dirs = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

    # We have csv for each epsilon value tried per image and then a csv with the resulting critical epsilon value per image
    df_epsilon = pd.DataFrame(columns = ['epsilon', 'result', 'runtime', 'network', 'image', 'ds'])
    df_epsilon_crit = pd.DataFrame(columns = ['image', 'Epsilon', 'runtime', 'network', 'ds'])

    for dir in dirs:
        NN_name = dir.split("\\")[-1]
        test_path = os.path.join(dir, "test")
        train_path = os.path.join(dir, "train")

        test_df_paths = os.listdir(test_path)
        train_df_paths = os.listdir(train_path)
        # remove distribution.csv
        test_df_paths.remove('distribution.csv')
        train_df_paths.remove('distribution.csv')

        # read in all test and train dfs
        for p in test_df_paths:
            df = pd.read_csv(os.path.join(test_path,p))
            df['network'] = NN_name
            df['ds'] = 'test'
            df['image'] = p.split("_")[-1].split(".")[0]
            df_epsilon = pd.concat([df_epsilon, df], ignore_index=True)

        for p in train_df_paths:
            df = pd.read_csv(os.path.join(train_path,p))
            df['network'] = NN_name
            df['ds'] = 'train'
            df['image'] = p.split("_")[-1].split(".")[0]
            df_epsilon = pd.concat([df_epsilon, df], ignore_index=True)

        # Now read in the critical epsilon values
        df = pd.read_csv(os.path.join(test_path, "distribution.csv"))
        df['network'] = NN_name
        df['ds'] = 'test'
        df_epsilon_crit = pd.concat([df_epsilon_crit, df], ignore_index=True)

        df = pd.read_csv(os.path.join(train_path, "distribution.csv"))
        df['network'] = NN_name
        df['ds'] = 'train'
        df_epsilon_crit = pd.concat([df_epsilon_crit, df], ignore_index=True)

    # Save the dataframe if it does not exist yet
    if not os.path.exists(os.path.join(path, "df_epsilon.csv")):
        df_epsilon.to_csv(os.path.join(path, "df_epsilon.csv"), index=False)

    if not os.path.exists(os.path.join(path, "df_epsilon_crit.csv")):
        df_epsilon_crit.to_csv(os.path.join(path, "df_epsilon_crit.csv"), index=False)
