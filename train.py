import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import wandb
import xgboost as xgb
from wandb.xgboost import wandb_callback


run = wandb.init(project="my-test-project")


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
y_train = df_train_split_preprocessed["Survived"]

X_test = df_test_split_preprocessed.drop(columns=['Survived'])
y_test = df_test_split_preprocessed["Survived"]


# WandB
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
# setup parameters for xgboost
# param = {}
# # use logistic binary classification
# param['objective'] = 'binary:logistic'
# # scale weight of positive examples
# param['eta'] = 0.1
# param['max_depth'] = 6
# param['silent'] = 1
# param['nthread'] = 4
# param['num_class'] = 2
param = {
        'objective': 'binary:logistic'
        , 'gamma': 1               ## def: 0
        , 'learning_rate': 0.1     ## def: 0.1
        , 'max_depth': 3
        , 'min_child_weight': 100  ## def: 1
        , 'n_estimators': 25
        , 'nthread': 24 
        , 'random_state': 42
        , 'reg_alpha': 0
        , 'reg_lambda': 0          ## def: 1
        , 'eval_metric': ['auc', 'logloss']
        , 'tree_method': 'hist'  # use gpu_hist to train on GPU
    }
#wandb.config.update(param)
run.config.update(dict(param))
xgbmodel = xgb.XGBClassifier(**param, use_label_encoder=False)
xgbmodel.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[wandb_callback()])
bstr = xgbmodel.get_booster()
# Save the booster to disk
model_name = f'{run.name}_model.json'
model_path = model_dir/model_name
bstr.save_model(str(model_path))

# Get the booster's config
config = json.loads(bstr.save_config())

# Log the trained model to W&B Artifacts, including the booster's config
model_art = wandb.Artifact(name=model_name, type='model', metadata=dict(config))
model_art.add_file(model_path)
run.log_artifact(model_art)


# watchlist = [(xg_train, 'train'), (xg_test, 'test')]
# num_round = 5
# bst = xgb.train(params=param, dtrain=xg_train, evals=watchlist, num_boost_round=num_round,callbacks=[wandb_callback()])
# #bst = xgb.train(param, xg_train, num_round, watchlist, callbacks=[wandb_callback()])
# # get prediction
# pred = bst.predict(xg_test)
# error_rate = np.sum(pred != y_test) / y_test.shape[0]
# print(f'Test error using logistic = {error_rate}')
# wandb.summary['Error Rate'] = error_rate