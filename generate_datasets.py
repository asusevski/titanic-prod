import pandas as pd

from sdv.tabular import CTGAN

df = pd.read_csv('./data/train.csv')

model = CTGAN()
model.fit(df)

n_people = len(df)
NUM_TITANICS = 10
for i in range(NUM_TITANICS):
    new_data = model.sample(num_rows=n_people)
    new_data.to_csv(f'./data/generated_data/generated_dataset{i}', index=False)
