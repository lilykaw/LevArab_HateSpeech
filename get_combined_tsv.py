import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path1', type=str)   # path to training data
parser.add_argument('--train_path2', type=str)   # path to training data
parser.add_argument('--test_path', type=str)     # path to test data
args = parser.parse_args()


# """ Create combined train tsv """
# ### Read in LHSAB and OSACT datasets
# with open(args.train_path1, 'r') as train_file_1:
#     df1 = pd.read_csv(train_file_1, sep="\t", header=None)                            # LHSAB train
# with open(args.train_path2, 'r') as train_file_2:
#     df2 = pd.read_csv(train_file_2, sep="\t", error_bad_lines=False, header=None)     # OSACT train
# newdf = df1.append(df2, ignore_index=True)

# ### Save as new tsv file
# newdf.to_csv('combined_train.tsv', sep='\t', header=False, index=False)


""" Create LHSAB tsv with binary labels """
### Standardize label column (for LHSAB dataset: "hate"->"HS", "normal"/"abusive"->"NOT_HS")
with open(args.test_path, 'r') as f:
    df = pd.read_csv(f, sep="\t", header=None)
df[1] = df[1].str.replace('hate', 'HS')
df[1] = df[1].str.replace('normal', 'NOT_HS')
df[1] = df[1].str.replace('abusive', 'NOT_HS')

### Save as new tsv file
df.to_csv('lhsab_test_binary.tsv', sep='\t', header=False, index=False)