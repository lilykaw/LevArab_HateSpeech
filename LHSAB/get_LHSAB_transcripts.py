from dataclasses import replace
import re
import csv
import random
from emot.emo_unicode import UNICODE_EMOJI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lhsab_tsv_path', type=str, required=True)    # path to input LHSAB tsv file
parser.add_argument('--output_train_path', type=str, required=True) # path to output train tsv file
parser.add_argument('--output_test_path', type=str, required=True)  # path to output test tsv file
args = parser.parse_args()


LHSAB_TSV_PATH = args.lhsab_tsv_path # '/Users/lilykawaoto/Documents/GitHub/LING-L715/L-HSAB.tsv'
LHSAB_OUTPUT_TRAIN_PATH = args.output_train_path
LHSAB_OUTPUT_TEST_PATH = args.output_test_path


def remove_emojis(txt):     # preprocessing
    text = ""
    for char in txt: 
        if char in UNICODE_EMOJI:
            continue
        else:
            text += char
    return text

def preprocess(txt):
    patterns = ['#', '@', 'USER', ':', ';', 'RT', 'URL', '<LF>', '\.\.\.', '…', '!', '\.', '\?', '%', '\*', '"', "'", '\$', '\&', '/', '\)', '\(', '\[', '\]', '\}', '\{', '|', '\d+']
    text = re.sub('|'.join(patterns), '', txt)          # remove patterns
    text = re.sub(r'[a-zA-Z]', '', text)                # remove non-Arabic characters
    text = re.sub(r'(.)\1\1+', r'\1', text)             # remove 3 or more repetitions of any character
    text = remove_emojis(text)                          # remove emojis
    return text

"""
Step 1: Read in tsv file. Preprocess each tweet as it's being read in. 
        Make train-test splits. Following Mulki et al. (2019), we then create the following sets: 
            - train set: 339 hate + 4337 non-hate (we combine abusive + normal) = 4676 total
            - test set:  129 hate + 1041 non-hate (we combine abusive + normal) = 1170 total
"""
train_hate, test_hate = [], []
train_non_hate, test_non_hate = [], []
with open(LHSAB_TSV_PATH, 'r') as f:
    next(f) # skip header
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        preproc_txt = preprocess(row[0])
        # print(preproc_txt)
        if row[1]=="hate":
            if len(train_hate) < 339:
                train_hate.append( (preproc_txt, row[1]) )
            else: 
                test_hate.append( (preproc_txt, row[1]) )
        elif row[1]=="abusive" or row[1]=="normal":
            if len(train_non_hate) < 4337:
                train_non_hate.append( (preproc_txt, row[1]) )
            else: 
                test_non_hate.append( (preproc_txt, row[1]) )
  
train_list = train_hate + train_non_hate
random.shuffle(train_list)
train_dict = dict(train_list)
with open(LHSAB_OUTPUT_TRAIN_PATH, 'w') as f:
    for key in train_dict.keys():
        f.write(f"{key}\t{train_dict[key]}\n")

test_list = test_hate + test_non_hate
random.shuffle(test_list)
test_dict = dict(test_list)
with open(LHSAB_OUTPUT_TEST_PATH, 'w') as f:
    for key in test_dict.keys():
        f.write(f"{key}\t{test_dict[key]}\n")
