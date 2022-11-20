import pandas as pd
from sdv.tabular import CTGAN


def main():
    # Read in real titanic data
    df = pd.read_csv('./data/train.csv')

    # Create CTGAN model and fit to preprocessed data
    model = CTGAN()
    model.fit(df)

    # Sample new datasets
    n_people = len(df)
    NUM_TITANICS = 10
    fractions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.9]

    for i, fraction in zip(range(NUM_TITANICS), fractions):
        new_data = model.sample(num_rows=n_people)

        # Randomly select some % of people and flip their survival to 1
        sample = new_data.sample(frac=fraction)
        sample['Survived'] = 1
        new_data.loc[sample.index] = sample.values

        # Save data
        new_data.to_csv(f'./data/generated_data2/generated_dataset{i}.csv', index=False)


if __name__ == "__main__":
    main()
    