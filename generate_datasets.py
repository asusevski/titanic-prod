import pandas as pd
from sdv.tabular import CTGAN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


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


def main():
    # Read in real titanic data
    df = pd.read_csv('./data/train.csv')

    pipe = Pipeline([
        ("dropper", Dropper()),
        ("imputer", Imputer()),
        ("feature_encoder", FeatureEncoder()),
        ("scaler", Scaler())
    ])

    df_preprocessed = pipe.fit_transform(df)

    # Create CTGAN model and fit to preprocessed data
    model = CTGAN()
    model.fit(df_preprocessed)

    # Sample new datasets
    n_people = len(df_preprocessed)
    NUM_TITANICS = 10
    for i in range(NUM_TITANICS):
        new_data = model.sample(num_rows=n_people)

        # Randomly select people from pclass3 (some small %) and with some probability p, flip their survival to 0

        new_data.to_csv(f'./data/generated_data/generated_dataset{i}.csv', index=False)


if __name__ == "__main__":
    main()
