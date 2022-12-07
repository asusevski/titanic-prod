import glob
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score, log_loss
import wandb
import xgboost as xgb


def predict_and_log_metrics(model, dataset, dataset_num, run):
    X = dataset.drop(columns=['Survived'])
    y_true = dataset['Survived'].values
    xg_data = xgb.DMatrix(X, label=y_true)
    y_pred = model.predict(xg_data)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    
    # Log stats
    run.summary[f'accuracy-{dataset_num}'] = accuracy_score(y_true, y_pred)
    run.summary[f'log_loss-{dataset_num}'] = log_loss(y_true, y_pred)


def plot_accuracy(run, num_datasets, title):
    accuracies = []
    for i in range(num_datasets):
        accuracy = run.summary[f'accuracy-{i}']
        accuracies.append(accuracy)
    
    titanics = list(range(1,11))
    data = [[titanic_num, accuracy] for (titanic_num, accuracy) in zip(titanics, accuracies)]
    table = wandb.Table(data=data, columns=["titanic_generation", "accuracy"])
    run.log({"Accuracies Over Time" : wandb.plot.line(table, "titanic_generation", "accuracy", title=title)})


def plot_log_loss(run, num_datasets, title):
    log_losses = []
    for i in range(num_datasets):
        log_loss = run.summary[f'log_loss-{i}']
        log_losses.append(log_loss)
    
    titanics = list(range(1,11))
    data = [[titanic_num, log_loss] for (titanic_num, log_loss) in zip(titanics, log_losses)]
    table = wandb.Table(data=data, columns=["titanic_generation", "log_loss"])
    run.log({"Log-Loss Over Time" : wandb.plot.line(table, "titanic_generation", "log_loss", title=title)})


def no_updating_model():
    run = wandb.init(project="titanic-no-updating-model")

    filepath_to_model = "./model/amber-meadow-40-model.json"

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
    
    df_train = pd.read_csv('./data/train_preprocessed.csv')
    for idx, df in enumerate(generated_datasets):
        # Predict on generated dataset
        predict_and_log_metrics(model, df, idx, run)

    # Visualize the accuracy and log-loss across the 10 titanics
    plot_accuracy(run, 10, "Accuracy over time (no updating model)")
    plot_log_loss(run, 10, "Log-Loss over time (no updating model)")
    wandb.finish()


def train_model(df):
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
    return booster


def update_model():
    run = wandb.init(project="titanic-updating-model")

    # Read train data
    df_train = pd.read_csv('./data/train_preprocessed.csv')
    df_train['trend'] = 0
    # Train model (need to re-train with new "Trend" variable)
    model = train_model(df_train)

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

    for idx, df in enumerate(generated_datasets):
        # Predict on data THEN update the model (simply re-training from scratch)
        predict_and_log_metrics(model, df, idx, run)

        # Add new dataset to training data
        df_train = pd.concat([df_train, df]).reset_index(drop=True)
        
        # Re-train model
        model = train_model(df_train)

    # Visualize the accuracy and log-loss across the 10 titanics
    plot_accuracy(run, 10, "Accuracy over time (with updating model)")
    plot_log_loss(run, 10, "Log-Loss over time (with updating model)")
    wandb.finish()
        

def main():
    no_updating_model()
    #update_model()
        

if __name__=="__main__":
    main()
