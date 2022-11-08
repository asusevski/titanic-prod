import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


df_train = pd.read_csv('./data/train.csv')
df_generated = pd.read_csv('./data/generated_data/generated_dataset0.csv')


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
y_test = df_test_split_preprocessed[["Survived"]]

# Generated data:
X_test_generated = df_generated.drop(columns=['Survived'])
y_test_generated = df_generated[["Survived"]]


# Training model
xg_model = xgb.XGBClassifier()

xg_model.fit(X_train,y_train)

testset_preds = xg_model.predict(X_test)
print(f"Accuracy on hold-out test set: {accuracy_score(y_test, testset_preds)}")



generated_preds = xg_model.predict(X_test_generated)
print(f"Accuracy on generated test set: {accuracy_score(y_test_generated, generated_preds)}")


print("===== Training on generated data =====")
xg_model = xgb.XGBClassifier()

xg_model.fit(X_test_generated, y_test_generated)

generated_preds_training = xg_model.predict(X_test_generated)

print(f"Training accuracy on generated data: {accuracy_score(y_test_generated, generated_preds_training)}")

testset_preds = xg_model.predict(X_test)
print(f"Accuracy on hold-out test set from real data: {accuracy_score(y_test, testset_preds)}")
