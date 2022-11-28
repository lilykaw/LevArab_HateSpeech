import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path1', type=str, required=True)   # path to training data
parser.add_argument('--train_path2', type=str, required=True)   # path to training data
args = parser.parse_args()


""" Create combined train tsv """
### Read in LHSAB and OSACT datasets
with open(args.train_path1, 'r') as train_file_1:
    df1 = pd.read_csv(train_file_1, sep="\t", header=None)                            # LHSAB train
with open(args.train_path2, 'r') as train_file_2:
    df2 = pd.read_csv(train_file_2, sep="\t", error_bad_lines=False, header=None)     # OSACT train
newdf = df1.append(df2, ignore_index=True)

### Standardize label column (for LHSAB dataset: "hate"->"HS", "normal"/"abusive"->"NOT_HS")
newdf[1] = newdf[1].str.replace('hate', 'HS')
newdf[1] = newdf[1].str.replace('normal', 'NOT_HS')
newdf[1] = newdf[1].str.replace('abusive', 'NOT_HS')

### Save as new tsv file
newdf.to_csv('combined_train.tsv', sep='\t', header=False, index=False)