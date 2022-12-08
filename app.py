import glob
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score, log_loss, f1_score
import wandb
import xgboost as xgb


def predict_and_log_metrics(model, dataset, dataset_num, run, updating_model):
    X = dataset.drop(columns=['Survived'])
    y_true = dataset['Survived'].values
    xg_data = xgb.DMatrix(X, label=y_true)
    y_pred = model.predict(xg_data)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    
    # Log stats
    if updating_model:
        run.summary[f'accuracy-updating-{dataset_num}'] = accuracy_score(y_true, y_pred)
        run.summary[f'log-loss-updating-{dataset_num}'] = log_loss(y_true, y_pred)
        run.summary[f'f1-score-updating-{dataset_num}'] = f1_score(y_true, y_pred)
    else:
        run.summary[f'accuracy-no-updating-{dataset_num}'] = accuracy_score(y_true, y_pred)
        run.summary[f'log-loss-no-updating-{dataset_num}'] = log_loss(y_true, y_pred)
        run.summary[f'f1-score-no-updating-{dataset_num}'] = f1_score(y_true, y_pred)


def plot_accuracy(run, num_datasets, title):
    accuracies_updating = []
    accuracies_no_updating = []
    for i in range(num_datasets):
        accuracy_updating = run.summary[f'accuracy-updating-{i}']
        accuracies_updating.append(accuracy_updating)

        accuracy_no_updating = run.summary[f'accuracy-no-updating-{i}']
        accuracies_no_updating.append(accuracy_no_updating)
    
    titanics = list(range(1, num_datasets + 1))
    data = [[titanic_num, acc_u, acc_no_u] for (titanic_num, acc_u, acc_no_u) in zip(titanics, accuracies_updating, accuracies_no_updating)]
    table = wandb.Table(data=data, columns=["titanic_generation", "accuracy_updating", "accuracy_without_updating"])
    run.log({"Accuracies Over Time" : wandb.plot.line_series(
        xs=titanics, ys=[accuracies_updating, accuracies_no_updating], 
        keys=["accuracy updating", "accuracy no updating"], title=title, xname="Titanic Generation")})


def plot_log_loss(run, num_datasets, title):
    log_losses_updating = []
    log_losses_no_updating = []
    for i in range(num_datasets):
        log_loss_updating = run.summary[f'log-loss-updating-{i}']
        log_losses_updating.append(log_loss_updating)

        log_loss_no_updating = run.summary[f'log-loss-no-updating-{i}']
        log_losses_no_updating.append(log_loss_no_updating)
    
    titanics = list(range(1, num_datasets + 1))
    data = [[titanic_num, logloss_u, logloss_no_u] for (titanic_num, logloss_u, logloss_no_u) in zip(titanics, log_losses_updating, log_losses_no_updating)]
    table = wandb.Table(data=data, columns=["titanic_generation", "log_loss_updating", "log_loss_without_updating"])
    run.log({"Log Losses Over Time" : wandb.plot.line_series(
        xs=titanics, ys=[log_losses_updating, log_losses_no_updating], 
        keys=["log-loss updating", "log-loss no updating"], title=title, xname="Titanic Generation")})


def plot_f1_score(run, num_datasets, title):
    f1_scores_updating = []
    f1_scores_no_updating = []
    for i in range(num_datasets):
        f1_score_updating = run.summary[f'f1-score-updating-{i}']
        f1_scores_updating.append(f1_score_updating)

        f1_score_no_updating = run.summary[f'f1-score-no-updating-{i}']
        f1_scores_no_updating.append(f1_score_no_updating)
    
    titanics = list(range(1, num_datasets + 1))
    data = [[titanic_num, f1_u, f1_no_u] for (titanic_num, f1_u, f1_no_u) in zip(titanics, f1_scores_updating, f1_scores_no_updating)]
    table = wandb.Table(data=data, columns=["titanic_generation", "f1_score_updating", "f1_score_without_updating"])
    run.log({"F1 Scores Over Time" : wandb.plot.line_series(
        xs=titanics, ys=[f1_scores_updating, f1_scores_no_updating], 
        keys=["f1-score updating", "f1-score no updating"], title=title, xname="Titanic Generation")})


def no_updating_model(run):
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
        predict_and_log_metrics(model, df, idx, run, False)


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


def update_model(run):

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
        predict_and_log_metrics(model, df, idx, run, True)

        # Add new dataset to training data
        df_train = pd.concat([df_train, df]).reset_index(drop=True)
        
        # Re-train model
        model = train_model(df_train)

    # Visualize the accuracy and log-loss across the 10 titanics
    # plot_accuracy(run, 10, "Accuracy over time (with updating model)")
    # plot_log_loss(run, 10, "Log-Loss over time (with updating model)")


def plot_updating_model_vs_no_updating_model(run):
    plot_accuracy(run, 10, "Plotting accuracy updating model vs no updating model")
    plot_log_loss(run, 10, "Plotting log-loss updating model vs no updating model")
    plot_f1_score(run, 10, "Plotting f1-score updating model vs no updating model")


def main():
    run = wandb.init(project="titanic-updating-vs-no-updating-model")
    update_model(run)
    no_updating_model(run)
    plot_updating_model_vs_no_updating_model(run)
    wandb.finish()
        

if __name__=="__main__":
    main()
