from dataclasses import replace
import argparse
import re
import csv
import random
import time
from googletrans import Translator # pip install googletrans==3.1.0a0
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO, UNICODE_EMOJI_ALIAS

parser = argparse.ArgumentParser()
parser.add_argument('--lev_path', type=str) # path to training data + labels
parser.add_argument('--other_paths', type=str, nargs='+') # paths to non-Levantine data (1 or more arguments)
args = parser.parse_args()

LEV_PATH = args.lev_path
OTHER_PATHS = args.other_paths 

def emoji_to_text(txt):     # helper for preprocess()
    # translator = google_translator()
    translator= Translator()
    text = ""
    for char in txt: 
        if char in UNICODE_EMOJI:
            tmp = char.replace(char, " ".join(UNICODE_EMOJI[char].replace(",","").replace(":","").split("_")))
            translation = translator.translate(tmp, dest='ar')
            text += "< " + translation.text + " >"
            # translation = translator.translate(tmp, lang_src='en', lang_tgt='ar')
            # text += "< " + translation + " >"
            text += " "
        elif char in UNICODE_EMOJI_ALIAS:
            tmp = char.replace(char, " ".join(UNICODE_EMOJI_ALIAS[char].replace(",","").replace(":","").split("_")))
            translation = translator.translate(tmp, dest='ar')
            text += "< " + translation.text + " >"
            # translation = translator.translate(tmp, lang_src='en', lang_tgt='ar')
            # text += "< " + translation + " >"
            text += " "
        else:
            text += char
    return text

def emoji_to_text2(txt):     # helper for preprocess()
    text = ""
    for char in txt: 
        if char in UNICODE_EMOJI or char in UNICODE_EMOJI_ALIAS:
            text += ""
        else:
            text += char
    return text

def emoticon_to_text(txt):     # helper for preprocess()
    translator= Translator()
    # translator = google_translator()
    text = ""
    for char in txt: 
        if char in EMOTICONS_EMO:
            tmp = char.replace(char, EMOTICONS_EMO[char])
            translation = translator.translate(tmp, dest='ar')
            # translation = translator.translate(tmp, lang_src='en', lang_tgt='ar')
            text += "< " + translation.text + " >"
            # text += "< " + translation + " >"
            text += " "
        else:
            text += char
    return text

def emoticon_to_text2(txt):     # helper for preprocess()
    text = ""
    for char in txt: 
        if char in EMOTICONS_EMO:
            text += ""
        else:
            text += char
    return text


def preprocess(txt):
    patterns = ['#', '@', 'USER', ':', ';', 'RT', 'URL', '<LF>', '\.\.\.', '…', '!', '\.', '\?', '%', '\*', '"', "'", '\$', '\&', '/', '\)', '\(', '\[', '\]', '\}', '\{', '|', '\d+', '_', '؟', 'ü', 'ı', 'ğ', 'ç']
    text = re.sub(r'@\w+ ', '', txt)                    # remove usernames
    text = re.sub('|'.join(patterns), '', text)         # remove patterns
    text = re.sub(r'[a-zA-Z0-9]', '', text)             # remove non-Arabic characters and numbers
    text = re.sub(r'\t', ' ', text)                     # replace tabs with single space
    text = re.sub(r'(.)\1\1+', r'\1', text)             # remove 3 or more repetitions of any character
    text = emoji_to_text2(text)                          # replace emojis with their Arabic description
    # time.sleep(0.5)
    text = emoticon_to_text2(text)                       # replace emoticons with their Arabic description
    # time.sleep(0.5)
    text = re.sub(r'http\S+', '', text)                 # remove URLs
    return text

"""
Read in csv file. Preprocess each tweet as it's being read in. 
Store cleaned text plus its label in a tsv file.
"""

""" TRAINING FILE"""
with open(LEV_PATH, 'r') as f:
   lev_text = []
   reader = csv.reader(f, delimiter="\t")
   for i,row in enumerate(reader): 
       row = [nonspace for nonspace in row if nonspace]
       assert(len(row)==3)
       lev_text.append(preprocess(row[2]))        
with open('clean_LEV.tsv', 'w') as f:
   for txt in lev_text:
       f.write(f"{txt}\tLEV\n")        


""" NON-LEVANTINE FILES """
text = []
for P in OTHER_PATHS:
    with open(P, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        for i,row in enumerate(reader): 
            row = [nonspace for nonspace in row if nonspace]
            assert(len(row)==3)
            text.append(preprocess(row[2]))
with open('clean_NONLEV.tsv', 'w') as f:
    for txt in text:
        f.write(f"{txt}\tNONLEV\n")


