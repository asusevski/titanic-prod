import glob
import numpy as np
import os
import pandas as pd
import re
from scipy.stats import kstest
from sklearn.metrics import accuracy_score, log_loss, f1_score
import xgboost as xgb



def predict_titanic(model, dataset):
    X = dataset.drop(columns=['Survived'])
    y_true = dataset['Survived'].values
    xg_data = xgb.DMatrix(X, label=y_true)
    y_pred = model.predict(xg_data)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    return y_pred


def print_metrics(y_pred, y_true):
    print(f"Count actually survivied: {y_true.sum()}")
    print(f"Count predicted survivied: {y_pred.sum()}")
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Log Loss: {log_loss(y_true, y_pred)}")

 
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


def drift_test_survival(y_latest, y_prev):
    #frac_surviving = y_prev.sum()
    #print(kstest(y_latest, stats..cdf))
    print(kstest(y_latest, y_prev)) # bernoulli?


def main():
    # Read in data:
    # Initial Titanic data
    df_original = pd.read_csv('./data/train_preprocessed.csv')
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

    # Initial titanic
    print("Training XgBoost Model on first Titanic data!")
    
    model = train_model(df_original)
    y_pred = predict_titanic(model, df_original)
    print("Metrics on original titanic dataset:")
    print_metrics(y_pred, df_original['Survived'].values)

    # Titanic_num stores the index of the upcoming titanic (the first one is the original one, so it is initialized to 2)
    titanic_num = 2
    while titanic_num < 12: # 11 total titanics
        print(f"Upcoming titanic iteration {titanic_num}")
        print(f"""
Select an option:
1. Predict next Titanic
2. Print metrics from last run (Titanic num {titanic_num-1})
3. Plot trend?
4. Train on all data
5. Test if the latest Titanic data was different than the second latest Titanic
        """)
        choice = input("> ")

        if choice not in ["1", "2", "3", "4", "5"]:
            print("Input not recognized. Try again.")
            continue
        
        if choice == "1":
            y_pred = predict_titanic(model, generated_datasets[titanic_num - 2])
            titanic_num += 1
        if choice == "2":
            if titanic_num == 2:
                print_metrics(y_pred, df_original['Survived'].values)
            else:
                print_metrics(y_pred, generated_datasets[titanic_num - 2]['Survived'].values)
        if choice == "3":
            continue
        if choice == "4":
            total_datasets = [df_train] + generated_datasets[:titanic_num - 2]
            df_cumulative = pd.concat(total_datasets)
            model = train_model(df_cumulative)
        if choice == "5":
            if titanic_num == 2:
                print("There's only been one titanic so far! Try again later.")
                continue
            if titanic_num == 3:
                y_prev = df_original['Survived'].values
                y_latest = generated_datasets[titanic_num - 2]['Survived'].values
                drift_test_survival(y_latest, y_prev)
            else:
                y_prev = generated_datasets[titanic_num - 3]['Survived'].values
                y_latest = generated_datasets[titanic_num - 2]['Survived'].values
                drift_test_survival(y_latest, y_prev)

        

if __name__=="__main__":
    main()
