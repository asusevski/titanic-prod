import glob
import os
import pandas as pd
import wandb
import xgboost as xgb


run = wandb.init(project="my-test-project")

filepath_to_model = "atomic-spaceship-36_model.json"

# Load model
model = xgb.Booster().load_model(filepath_to_model)

# Read in generated data
path = './data/generated_data'
generated_files = glob.glob(os.path.join(path , "/*.csv"))

generated_datasets = []

for filename in generated_files:
    df = pd.read_csv(filename)
    generated_datasets.append(df)
