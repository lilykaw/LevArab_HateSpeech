import pandas as pd

COMBINED_TSV_PATH = 'combined_train.tsv'

### Read in LHSAB and OSACT datasets
df1 = pd.read_csv('/Users/lilykawaoto/Documents/GitHub/LING-L715/lhsab_train.tsv', sep="\t", header=None)             # LHSAB train
df2 = pd.read_csv('/Users/lilykawaoto/Documents/GitHub/LING-L715/osact_train_cleaned.tsv', sep="\t", error_bad_lines=False, header=None)     # OSACT train
newdf = df1.append(df2, ignore_index=True)

### Standardize label column (for LHSAB dataset: "hate"->"HS", "normal"/"abusive"->"NOT_HS")
newdf[1] = newdf[1].str.replace('hate', 'HS')
newdf[1] = newdf[1].str.replace('normal', 'NOT_HS')
newdf[1] = newdf[1].str.replace('abusive', 'NOT_HS')

### Save as new tsv file
newdf.to_csv('combined_train.tsv', sep='\t', header=False, index=False)
