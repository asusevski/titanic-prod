<<<<<<< HEAD
import json
import numpy as np
=======
>>>>>>> 369f11583313b7654913da4e81fffe22f2c98021
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
<<<<<<< HEAD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
=======
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
>>>>>>> 369f11583313b7654913da4e81fffe22f2c98021
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import wandb
import xgboost as xgb
from wandb.integration.xgboost import WandbCallback


run = wandb.init(project="my-test-project-2")


df_train = pd.read_csv('./data/train.csv')


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5)
for train_index, test_index in split.split(df_train, df_train["Pclass"]):
    train_split = df_train.loc[train_index]
    test_split = df_train.loc[test_index]


class Dropper(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])


class Imputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        imputer = SimpleImputer(strategy="median")
        X['Age'] = imputer.fit_transform(X[['Age']])
        X['Fare'] = imputer.fit_transform(X[['Fare']])
        
        X['Embarked'] = X['Embarked'].fillna('S')
        return X


class FeatureEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # One-hot encodings:
        # Note that we'll have to drop the categorical cols at the end
        
        # Encoding Sex:
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        # Encoding Female into a one-hot vector
        X[encoder.categories_[0][0]] = matrix[:, 0]
        
        # Encoding Pclass:
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Pclass']]).toarray()
        # Encoding Pclass 1 and 2 into one-hot vectors
        for idx, col in enumerate(encoder.categories_[0][:-1]):
            X["Pclass" + str(col)] = matrix[:, idx]
            
        # Encoding Embarked:
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()
        # Encoding Pclass 1 and 2 into one-hot vectors
        for idx, col in enumerate(encoder.categories_[0][:-1]):
            X[col] = matrix[:, idx]
            
        return X.drop(columns=['Sex', 'Pclass', 'Embarked'])


class Scaler(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        scaler = StandardScaler()
        X['Age'] = scaler.fit_transform(X[['Age']])
        
        X['Fare'] = scaler.fit_transform(X[['Fare']])
        return X


pipe = Pipeline([
    ("dropper", Dropper()),
    ("imputer", Imputer()),
    ("feature_encoder", FeatureEncoder()),
    ("scaler", Scaler())
])


# Preprocessing:
df_train_split_preprocessed = pipe.fit_transform(train_split)
df_test_split_preprocessed = pipe.fit_transform(test_split)

X_train = df_train_split_preprocessed.drop(columns=['Survived'])
y_train = df_train_split_preprocessed[["Survived"]]

X_test = df_test_split_preprocessed.drop(columns=['Survived'])
<<<<<<< HEAD
y_test = df_test_split_preprocessed["Survived"]


# WandB
# Create a W&B Table and log 50 random rows of the dataset to explore
train_table = wandb.Table(dataframe=df_train_split_preprocessed.sample(50))
test_table = wandb.Table(dataframe=df_test_split_preprocessed.sample(50))

# Log the Table to your W&B workspace
wandb.log({'processed_train_dataset': train_table})
wandb.log({'processed_test_dataset': test_table})


xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
param = {
        'objective': 'binary:logistic',
        'gamma': 1,  
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 100,
        'n_estimators': 25,
        'nthread': 24,
        'random_state': 42,
        'reg_alpha': 0,
        'reg_lambda': 0,         
        'eval_metric': ['auc', 'logloss'],
        'tree_method': 'hist', 
        'callbacks' : [wandb_callback()]
}

run.config.update(dict(param))

# Training
booster = xgb.train(
    {'objective': 'binary:logistic',
     'eval_metric': ['auc', 'logloss'],
     'tree_method': 'hist'
     }, xg_train,
    evals=[(xg_train, 'Train'), (xg_test, 'Test')],
    num_boost_round=50
)


# Saving and logging info 
# Save the booster to disk
model_name = f'{run.name}-model.json'
model_dir = "./model"
model_path = f"{model_dir}/{model_name}"
booster.save_model(str(model_path))

# Get the booster's config
config = json.loads(booster.save_config())

# Log the trained model to W&B Artifacts, including the booster's config
model_art = wandb.Artifact(name=model_name, type='model', metadata=dict(config))
model_art.add_file(model_path)
run.log_artifact(model_art)

# Add the additional data from the booster's config to the run config
run.config.update(dict(config))

# Get train and validation predictions
y_pred_train = booster.predict(xg_train)
y_pred_test = booster.predict(xg_test)

# Log additional Train metrics
y_pred_test_classes = np.where(y_pred_test >= 0.5, 1, 0)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_pred_train) 
run.summary['train_ks_stat'] = max(true_positive_rate - false_positive_rate)
run.summary['train_auc'] = auc(false_positive_rate, true_positive_rate)
run.summary['train_log_loss'] = -(y_train * np.log(y_pred_train) + (1-y_train) * np.log(1-y_pred_train)).sum() / len(y_train)

# Log additional Validation metrics
run.summary["val_auc"] = roc_auc_score(y_test, y_pred_test)
run.summary["val_acc_0.5"] = accuracy_score(y_test, y_pred_test_classes)
run.summary["val_log_loss"] = -(y_test * np.log(y_pred_test) + (1-y_test) * np.log(1-y_pred_test)).sum() / len(y_test)

y_test_preds_2d = np.array([1-y_pred_test, y_pred_test])  # W&B expects a 2d array
y_test_arr = y_test.values
d = 0
while len(y_test_preds_2d.T) > 10000:
    d +=1
    y_test_preds_2d = y_test_preds_2d[::1, ::d]
    y_test_arr = y_test_arr[::d]
run.log({"ROC_Curve" : wandb.plot.roc_curve(y_test_arr, y_test_preds_2d.T,
                                           labels=['did not survive','survived'],
                                           classes_to_plot=[1])})

# Log Feature Importance
fi = booster.get_fscore()
fi_data = [[k, fi[k]] for k in fi]
table = wandb.Table(data=fi_data, columns = ["Feature", "Importance"])
run.log({"Feature Importance" : wandb.plot.bar(table, "Feature",
                               "Importance", title="Feature Importance")})
=======
y_test = df_test_split_preprocessed[["Survived"]]


config = wandb.config
config.seed = 123

config.test_size = 0.2
config.colsample_bytree = 0.3
config.learning_rate = 0.01
config.max_depth = 15
config.alpha = 10
config.n_estimators = 5

wandb.config.update(config)

xg_model = xgb.XGBClassifier(objective='binary:logistic', 
    colsample_bytree=config.colsample_bytree, 
    learning_rate=config.learning_rate,
    max_depth=config.max_depth, 
    alpha=config.alpha, 
    n_estimators=config.n_estimators)

xg_model.fit(X_train,y_train,
           callbacks=[WandbCallback()]
)

preds = xg_model.predict(X_test)

run.summary["test_score"] = accuracy_score(y_test, preds)

wandb.finish()
>>>>>>> 369f11583313b7654913da4e81fffe22f2c98021
