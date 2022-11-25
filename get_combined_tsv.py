import pandas as pd


df1 = pd.read_csv('/Users/lilykawaoto/Documents/GitHub/LING-L715/lhsab_train.tsv', sep="\t")              # LHSAB train
df2 = pd.read_csv('/Users/lilykawaoto/Documents/GitHub/LING-L715/osact_train_cleaned.tsv', sep="\t")      # OSACT train
newdf = df1.merge(df2, how='outer')
newdf.to_csv('/Users/lilykawaoto/Documents/GitHub/LING-L715/combined_train.tsv', sep="\t")