from dataclasses import replace
import os, sys, re
import csv
import string
import pickle
# from translate import Translator # limit to how many times you can use this in a day lmao
from googletrans import Translator, constants
from emot.emo_unicode import UNICODE_EMOJI


CV_TSV_PATH = '/Users/lilykawaoto/Documents/GitHub/L715 - Hate Speech on Levantine Tweets/L-HSAB.tsv'


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
with open(CV_TSV_PATH, 'r') as f:
    next(f) # skip header
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        preproc_txt = emoji_to_text(row[0])
        lshab_dict[preproc_txt] = row[1]
        if row[1]=="hate":
            hate.append(preproc_txt)
        elif row[1]=="abusive":
            abusive.append(preproc_txt)
        elif row[1]=="normal":
            normal.append(preproc_txt)
# print(lshab_dict.keys())
print(len(hate))
print(len(abusive))
print(len(normal))