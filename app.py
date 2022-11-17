import glob
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score
import wandb
import xgboost as xgb


#run = wandb.init(project="my-test-project")

filepath_to_model = "./model/atomic-spaceship-36_model.json"

# Load model
model = xgb.Booster()
model.load_model("model/charmed-firefly-38-model.json")

# Read in generated data
path = './data/generated_data/'
generated_files = glob.glob(os.path.join(path , "*.csv"))

generated_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

generated_datasets = []
for filename in generated_files:
    df = pd.read_csv(filename)
    generated_datasets.append(df)


for df in generated_datasets:
    xg_data = xgb.DMatrix(df.drop(columns=['Survived']), label=df['Survived'])
    y_pred = model.predict(xg_data)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    print(accuracy_score(df['Survived'].values, y_pred))

