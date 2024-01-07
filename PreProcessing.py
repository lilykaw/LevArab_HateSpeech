!pip install tashaphyne
import os
import re
from tashaphyne.normalize import strip_tashkeel, strip_tatweel, normalize_hamza, normalize_lamalef, normalize_spellerrors

def kashida_removal(text):
    text = re.sub(r'ـ+', '', text)
    return text

def arabic_diacritics_removal(text):
    diacritics = 'ًٌٍَُِّّْٰ'
    for d in diacritics:
        text = text.replace(d, '')
    return text

def remove_punctuation(text):
    punctuations = r"""!"#$%&'()*+,-./:;<=>?@[\]^_‘;`{|}~•«»…“”–—٪"""
    arabic_punctuations = r"؟،؛ـ"
    all_punctuations = punctuations + arabic_punctuations
    translator = str.maketrans('', '', all_punctuations)
    return text.translate(translator)

def alef_lam_normalization(text):
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'[ؤئ]', 'ء', text)
    text = re.sub(r'[ةه]', 'ه', text)
    return text

def normalize_text(text):
    new_text = strip_tashkeel(text)
    new_text = strip_tatweel(new_text)
    new_text = normalize_hamza(new_text)
    new_text = normalize_lamalef(new_text)
    new_text = arabic_diacritics_removal(new_text)
    new_text = alef_lam_normalization(new_text)
    new_text = kashida_removal(new_text)
    new_text = remove_punctuation(new_text)
    new_text = normalize_spellerrors(new_text)
    return new_text

def normalize_tweets(new_text):
    RemoveHTML = re.sub(r'http.*', '', new_text)
    RemoveRT = re.sub(r'RT ', '', RemoveHTML)
    RemoveU = re.sub(r'@\w+', '', RemoveRT)
    RemoveHash = re.sub(r'#\w+', '', RemoveU)
    NoSpace = re.sub(r'[\s]{2,}', ' ', RemoveHash)
    return NoSpace

def normalize_repeated_letters(NoSpace):
    # Find all occurrences of repeated Arabic letters (3 times or more)
    repeated_letters_pattern = re.compile(r'([ء-ي])\1{2,}', re.UNICODE)

    # Replace repeated letters with a single occurrence
    matches = re.finditer(repeated_letters_pattern, NoSpace)
    for match in matches:
        repeated_string = match.group(0)
        normalized_string = match.group(1)
        NoSpace = NoSpace.replace(repeated_string, normalized_string)

    return NoSpace

# Specify the working directory
directory = os.getcwd()

# Specify the input file name
input_file = "MoreHate.tsv"

# Construct the file path
filepath = os.path.join(directory, input_file)

# Read the content of the input file
with open(filepath, mode='r', encoding='utf-8') as file:
    text = file.read()

# Apply the normalization functions
normalized_text = normalize_text(text)
normalized_text = normalize_tweets(normalized_text)
normalized_text = normalize_repeated_letters(normalized_text)

# Write the normalized text back to the input file
with open(filepath, mode='w', encoding='utf-8') as file:
    file.write(normalized_text)
