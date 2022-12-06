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
# parser.add_argument('--Xtest_path', type=str)   # path to test data
# parser.add_argument('--Ytest_path', type=str)   # path to test labels
args = parser.parse_args()

LEV_PATH = args.lev_path 
# OSACT_XYTRAIN_PATH = args.XYtrain_path  # '/Users/lilykawaoto/Documents/GitHub/LING-L715/OSACT/OSACT_train.csv'
# OSACT_XTEST_PATH = args.Xtest_path      # '/Users/lilykawaoto/Documents/GitHub/LING-L715/OSACT/OSACT2020-sharedTask-CodaLab-Train-Dev-Test/OSACT2020-sharedTask-test-tweets.txt'
# OSACT_YTEST_PATH = args.Ytest_path      # '/Users/lilykawaoto/Documents/GitHub/LING-L715/OSACT/OSACT2020-sharedTask-CodaLab-Train-Dev-Test/OSACT2020-sharedTask-test-taskB-gold-labels.txt'

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


def preprocess(txt):
    patterns = ['#', '@', 'USER', ':', ';', 'RT', 'URL', '<LF>', '\.\.\.', '…', '!', '\.', '\?', '%', '\*', '"', "'", '\$', '\&', '/', '\)', '\(', '\[', '\]', '\}', '\{', '|', '\d+']
    text = re.sub(r'@\w+ ', '', txt)                    # remove usernames
    text = re.sub('|'.join(patterns), '', text)         # remove patterns
    text = re.sub(r'[a-zA-Z]', '', text)                # remove non-Arabic characters
    text = re.sub(r'\t', ' ', text)                     # replace tabs with single space
    text = re.sub(r'(.)\1\1+', r'\1', text)             # remove 3 or more repetitions of any character
    text = emoji_to_text(text)                          # replace emojis with their Arabic description
    time.sleep(0.5)
    text = emoticon_to_text(text)                       # replace emoticons with their Arabic description
    time.sleep(0.5)
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
    print(lev_text)
        
        
# with open('osact_train_cleaned4.tsv', 'w') as f:
#     for pair in osact_train_list:
#         f.write(f"{pair[0]}\t{pair[1]}\n")


# """ TEST FILES (TXT & LABELS) """
# with open(OSACT_XTEST_PATH, 'r') as f1, open(OSACT_YTEST_PATH, 'r') as f2:
#     reader1 = f1.read().splitlines() 
#     reader2 = f2.read().splitlines()
#     osact_test_list = [(preprocess(row), reader2[i]) for i,row in enumerate(reader1)]  
# with open('osact_test_cleaned3.tsv', 'w') as f:
#     for pair in osact_test_list:
#         f.write(f"{pair[0]}\t{pair[1]}\n")