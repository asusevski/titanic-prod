import glob
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score
import wandb
import xgboost as xgb


def predict_and_log_metrics(model, dataset, run):
    X = dataset.drop(columns=['Survived'])
    y_true = dataset['Survived'].values
    xg_data = xgb.DMatrix(X, label=y_true)
    y_pred = model.predict(xg_data)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print(accuracy_score(y_true, y_pred))
    
    run.summary['train_log_loss'] = -(y_train * np.log(y_pred_train) + (1-y_train) * np.log(1-y_pred_train)).sum() / len(y_train)


def no_updating_model():
    #run = wandb.init(project="my-test-project")

    filepath_to_model = "./model/charmed-firefly-38-model.json"

    # Load model
    model = xgb.Booster()
    model.load_model(filepath_to_model)

    # Read in generated data
    path = './data/generated_data/'
    generated_files = glob.glob(os.path.join(path , "*.csv"))

    # Sort the files so that the order is generated_dataset0, generated_dataset1, generated_dataset2 etc... 
    generated_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Load generated datasets into dataframes
    generated_datasets = []
    for filename in generated_files:
        df = pd.read_csv(filename)
        generated_datasets.append(df)
    
    for df in generated_datasets:
        predict_and_log_metrics(model, df)


def train_model_and_log_metrics(df):
    X = df.drop(columns=['Survived']).values
    y = df['Survived'].values

    xg_train = xgb.DMatrix(X, label=y)

    booster = xgb.train(
        {'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'tree_method': 'hist'
        }, xg_train,
        evals=[(xg_train, 'Train')],
        num_boost_round=50
    )

    predict_and_log_metrics(booster, df)


def update_model():
    #run = wandb.init(project="my-test-project")

    filepath_to_model = "./model/charmed-firefly-38-model.json"

    # Load model
    model = xgb.Booster()
    model.load_model(filepath_to_model)

    # Read in generated data
    path = './data/generated_data/'
    generated_files = glob.glob(os.path.join(path , "*.csv"))

    # Sort the files so that the order is generated_dataset0, generated_dataset1, generated_dataset2 etc... 
    generated_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Load generated datasets into dataframes
    generated_datasets = []
    for idx, filename in enumerate(generated_files):
        df = pd.read_csv(filename)

        # Add trend (offset of 1 since the real data will have trend=0)
        df['trend'] = idx + 1
        generated_datasets.append(df)
    
    df_train = pd.read_csv('./data/train_preprocessed.csv')
    df_train['trend'] = 0
    for df in generated_datasets:
        df_train = pd.concat([df_train, df]).reset_index(drop=True)
        train_model_and_log_metrics(df_train)
        

def main():
    #no_updating_model()
    update_model()

    # #run = wandb.init(project="my-test-project")

    # filepath_to_model = "./model/charmed-firefly-38-model.json"

    # # Load model
    # model = xgb.Booster()
    # model.load_model(filepath_to_model)

    # # Read in generated data
    # path = './data/generated_data/'
    # generated_files = glob.glob(os.path.join(path , "*.csv"))

    # # Sort the files so that the order is generated_dataset0, generated_dataset1, generated_dataset2 etc... 
    # generated_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # # Load generated datasets into dataframes
    # generated_datasets = []
    # for filename in generated_files:
    #     df = pd.read_csv(filename)
    #     generated_datasets.append(df)
    
    # for df in generated_datasets:
    #     predict_and_log_metrics(model, df)
        

if __name__=="__main__":
    main()
