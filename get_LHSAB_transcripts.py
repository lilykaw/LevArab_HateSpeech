from dataclasses import replace
import os, sys, re
import csv
import string
import random
import pickle
# from translate import Translator # limit to how many times you can use this in a day lmao
from googletrans import Translator, constants
from emot.emo_unicode import UNICODE_EMOJI


LHSAB_TSV_PATH = '/Users/lilykawaoto/Documents/GitHub/L715 - Hate Speech on Levantine Tweets/L-HSAB.tsv'


def emoji_to_text(txt):     # preprocessing
    translator= Translator()
    text = ""
    for char in txt: 
        if char in UNICODE_EMOJI:
            tmp = char.replace(char, " ".join(UNICODE_EMOJI[char].replace(",","").replace(":","").split("_")))
            
            translation = translator.translate(tmp, dest='ar')
            text += "< " + translation.text + " >"
            text += " "
        else:
            text += char
    return text

"""
Step 1: read in tsv file. preprocess each tweet as it's being read in. store transcripts + labels in a dict.
"""
lshab_dict = {}
hate = []
abusive = []
normal = []
with open(LHSAB_TSV_PATH, 'r') as f:
    next(f) # skip header
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        preproc_txt = emoji_to_text(row[0])
        # print(preproc_txt)
        lshab_dict[preproc_txt] = row[1]
        if row[1]=="hate":
            hate.append(preproc_txt)
        elif row[1]=="abusive":
            abusive.append(preproc_txt)
        elif row[1]=="normal":
            normal.append(preproc_txt)

"""
Step 2: make train-test splits. Following Mulki et al. (2019), we then create the following sets: 
    - train set: 339 hate + 4337 non-hate (we combine abusive + normal) = 4676 total
    - test set:  129 hate + 1041 non-hate (we combine abusive + normal) = 1170 total
Each set had been randomized before being divided into train/test splits.
"""

shuffled_hate = random.shuffle(hate)
shuffled_nonhate = random.shuffle(abusive+normal)
train_set, test_set = {}, {}
for i in range(339):
    train_set[shuffled_hate[i]] = 1
for i in range(4337):
    train_set[shuffled_nonhate[i]] = 0

train_set = random.shuffle( shuffled_hate[:339] + shuffled_nonhate[:4337] )
test_set = random.shuffle( shuffled_hate[:339] + shuffled_nonhate[:4337] )
