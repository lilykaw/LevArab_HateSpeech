from dataclasses import replace
import os, sys, re
import csv
import string
import pickle
# from translate import Translator # limit to how many times you can use this in a day lmao
from googletrans import Translator, constants
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO, UNICODE_EMOJI_ALIAS


OSACT_TSV_PATH = '/Users/lilykawaoto/Documents/GitHub/L715 - Hate Speech on Levantine Tweets/OSACT2020-sharedTask-CodaLab-Train-Dev-Test/OSACT2020-sharedTask-train.tsv'

def emoji_to_text(txt):     # helper for preprocess()
    translator= Translator()
    text = ""
    for char in txt: 
        if char in UNICODE_EMOJI:
            tmp = char.replace(char, " ".join(UNICODE_EMOJI[char].replace(",","").replace(":","").split("_")))
            translation = translator.translate(tmp, dest='ar')
            text += "< " + translation.text + " >"
            text += " "
        elif char in UNICODE_EMOJI_ALIAS:
            tmp = char.replace(char, " ".join(UNICODE_EMOJI_ALIAS[char].replace(",","").replace(":","").split("_")))
            translation = translator.translate(tmp, dest='ar')
            text += "< " + translation.text + " >"
            text += " "
        else:
            text += char
    return text

def emoticon_to_text(txt):     # helper for preprocess()
    translator= Translator()
    text = ""
    for char in txt: 
        if char in EMOTICONS_EMO:
            tmp = char.replace(char, EMOTICONS_EMO[char])
            translation = translator.translate(tmp, dest='ar')
            text += "< " + translation.text + " >"
            text += " "
        else:
            text += char
    return text


def preprocess(txt):
    patterns = ['#', '@', 'USER', ':', ';', 'RT', 'URL', '<LF>', '\.\.\.', '…', '!', '\.', '\?', '%', '\*', '"', "'", '\$', '\&', '/', '\)', '\(', '\[', '\]', '\}', '\{', '|', '\d+']
    text = re.sub('|'.join(patterns), '', txt)          # remove patterns
    text = re.sub(r'[a-zA-Z]', '', text)                # remove non-Arabic characters
    text = re.sub(r'(.)\1\1+', r'\1', text)             # remove 3 or more repetitions of any character
    
    # text = re.sub("_", " ", text)       # replace underscore with space
    text = emoji_to_text(text)                          # repalce emojis with their Arabic description
    text = emoticon_to_text(text)                       # replace emoticons with their Arabic description
    ## text = normalize_dialect(txt)                    # normalize dialect
    return text

"""
Step 1: read in tsv file. preprocess each tweet as it's being read in. store transcripts + labels in a dict.
"""
osact_dict = {}
hate = []
not_hate = []
with open(OSACT_TSV_PATH, 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        preproc_txt = preprocess(row[0])
        print(preproc_txt)
        osact_dict[preproc_txt] = row[2]
        if row[2]=="HS":
            hate.append(preproc_txt)
        elif row[2]=="NOT_HS":
            not_hate.append(preproc_txt)