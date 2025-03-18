import pandas as pd

def sample_data(input_file, output_file, sample_fraction=0.2, random_state=None):
    df = pd.read_csv(input_file)

    sampled_df = df.sample(frac=sample_fraction, random_state=random_state)

    output_df = sampled_df[['ID', 'TEXT', 'LABEL']]

    output_df.to_csv(output_file, index=False)

    print(f"Sampled {len(output_df)} rows from {input_file} and saved to {output_file}.")

input_file = "data/train.csv"
output_file = "data/held_out.csv"

sample_data(input_file, output_file)
