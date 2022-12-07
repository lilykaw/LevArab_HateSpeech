import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--clean_lev_path', type=str)       # path to LEV data      (e.g. "clean_LEV.tsv")
parser.add_argument('--clean_nonlev_path', type=str)    # path to non-LEV data  (e.g. "clean_NONLEV.tsv") 
args = parser.parse_args()

"""
Create train-test splits of Levantine & non-Levantine data.
"""
with open(args.clean_lev_path, 'r') as f1:
    lev = pd.read_csv(f1, delimiter="\t", ignore_index=True)
with open(args.clean_nonlev_path, 'r') as f2:
    nonlev = pd.read_csv(f2, delimiter="\t", ignore_index=True)

data = pd.concat([lev, nonlev], axis=0)
train_df, test_df = train_test_split(data, shuffle=True, test_size=0.2)  # shuffle & use 80train/20test


"""
Set up the LSTM classifier.

>>Why LSTM? See paper by Lulu & Elnagar (2019):
Lulu, L., & Elnagar, A. (2018). Automatic Arabic dialect classification using deep learning models. Procedia computer science, 142, 262-269.
"""

# include LSTM code here -- reference rules.ipynb 