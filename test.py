import pandas as pd

with open("/Users/lilykawaoto/Documents/GitHub/LING-L715/OSACT/osact_train_cleaned.tsv", 'r') as f:
    osact_df = pd.read_csv(f, delimiter="\t", header=None, index_col=False, error_bad_lines=False)
    osact_df = osact_df.dropna()

osact_tweets = osact_df.iloc[:, 0].tolist()
osact_labels = osact_df.iloc[:, 1].tolist()