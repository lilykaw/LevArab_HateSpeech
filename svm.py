import argparse
from cmath import isnan
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report


"""
Step 1: read in input args
"""
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True)    # path to training data
parser.add_argument('--test_path', type=str, required=True)     # path to test data 
# parser.add_argument('--output_path', type=str, required=True)   # path to output file (metric results)
args = parser.parse_args()

with open(args.train_path, 'r') as train_f:
    col_names = ["text", "label"]
    train_df = pd.read_csv(train_f, delimiter="\t", names=col_names)
    train_df.dropna()
    # df.head()
    x_train = [row[0] for row in train_df.itertuples(index=False)]
    y_train = [row[1] for row in train_df.itertuples(index=False)]

with open(args.test_path, 'r') as test_f:
    col_names = ["text", "label"]
    test_df = pd.read_csv(test_f, delimiter="\t", names=col_names)
    test_df.dropna()
    x_test = [row[0] for row in test_df.itertuples(index=False)]
    y_test = [row[1] for row in test_df.itertuples(index=False)]


"""
Step 2: create unigrams, unigrams+bigrams, unigrams+bigrams+trigrams
        create term-frequency ratings with thresholds of 2 & 3   // term-freq: the relative freq of a term t within a document d
        returns a dict of n-grams and each item's frequecncy 
"""
# creates a dictionary of ngrams
def make_ngram(n, threshold, txt_list):
    ngrams_dict = {}
    # ngrams_list = []
    for txt in txt_list:
        if pd.isna(txt): # TO-DO: debug to figure out why there are NaN values in txt_list
            continue
        if n==1:        # make unigrams
            words = txt.split(" ")
            for word in words:
                # ngrams_list.append(word)
                if word not in ngrams_dict:
                    ngrams_dict[word] = 1
                else:
                    ngrams_dict[word] += 1

        elif n==2:      # make bigrams
            words = txt.split(" ")
            for i,word in enumerate(words):
                if i==0:
                    continue
                else:
                    # ngrams_list.append((txt[i-1], word))
                    if (words[i-1], word) not in ngrams_dict:
                        ngrams_dict[(words[i-1], word)] = 1
                    else:
                        ngrams_dict[(words[i-1], word)] += 1

        elif n==3:      # make trigrams
            words = txt.split(" ")
            if len(words) < 3:
                continue
            for i,word in enumerate(words):
                if i<2:
                    continue
                else: 
                    # ngrams_list.append((txt[i-2], txt[i-1], word))
                    if (words[i-2], words[i-1], word) not in ngrams_dict:
                        ngrams_dict[(words[i-2], words[i-1], word)] = 1
                    else: 
                        ngrams_dict[(words[i-2], words[i-1], word)] += 1

    assert(threshold==2 or threshold==3)
    if threshold==2:
        ngrams_dict = {key:val for key, val in ngrams_dict.items() if val>2}

    elif threshold==3:
        ngrams_dict = {key:val for key, val in ngrams_dict.items() if val>3}

    return ngrams_dict


# returns input text data, vectorized according to ngram_list
#       @text: input text
#       @ngram_list: list of ngrams
#       @vectorized_list: dictionary of token:index, where len(index)==vocab size
def make_vectorized_data(text, ngram_list, vectorized_list):
    X = []
    for txt in text:
        split_txt = str(txt).split(" ")
        tmp = [0] * len(ngram_list)
        for word in split_txt:
            if word in ngram_list:
                tmp[vectorized_list[word]] += 1
        X.append(tmp)
    return X


# trains SVM on X_train & y_train; tests X_test and y_test as y_pred
# also prints out Accuracy, Precision, Recall, & F1-score
def train_test(X_train, y_train, X_test, y_test):
    clf = svm.SVC(kernel='linear')
    print("Training...")
    clf.fit(X_train, y_train)
    print("Testing...")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"Precision: {metrics.precision_score(y_test, y_pred, labels=['abusive', 'hate', 'normal'], average='macro')}")
    print(f"Recall: {metrics.recall_score(y_test, y_pred, labels=['abusive', 'hate', 'normal'], average='macro')}")
    print(f"Macro-f1: {metrics.f1_score(y_test, y_pred, labels=['abusive', 'hate', 'normal'], average='macro')}")
    print(classification_report(y_test, y_pred))



# """
# Step 3: train & predict
#         Linear SVM with default parameters
# """

##### threshold == 2

## unigrams 
print("\n===== threshold 2, unigrams =====")
unigrams_2 = list(make_ngram(1, 2, x_train).keys())
vectorized_uni_2 = {key:ind for ind,key in enumerate(unigrams_2)}
X_train = np.array(make_vectorized_data(x_train, unigrams_2, vectorized_uni_2))
X_test = np.array(make_vectorized_data(x_test, unigrams_2, vectorized_uni_2))
train_test(X_train, y_train, X_test, y_test)

## unigrams + bigrams
print("\n===== threshold 2, unigrams + bigrams =====")
bigrams_2 = list(make_ngram(2, 2, x_train).keys())
uni_bi_2 = unigrams_2 + bigrams_2
vectorized_uni_bi_2 = {key:ind for ind,key in enumerate(uni_bi_2)}
X_train = np.array(make_vectorized_data(x_train, uni_bi_2, vectorized_uni_bi_2))
X_test = np.array(make_vectorized_data(x_test, uni_bi_2, vectorized_uni_bi_2))
train_test(X_train, y_train, X_test, y_test)

## unigrams + bigrams + trigrams  
print("\n===== threshold 2, unigrams + bigrams + trigrams =====")   
trigrams_2 = list(make_ngram(3, 2, x_train).keys())
uni_bi_tri_2 = uni_bi_2 + trigrams_2
vectorized_uni_bi_tri_2 = {key:ind for ind,key in enumerate(uni_bi_tri_2)}
X_train = np.array(make_vectorized_data(x_train, uni_bi_tri_2, vectorized_uni_bi_tri_2))
X_test = np.array(make_vectorized_data(x_test, uni_bi_tri_2, vectorized_uni_bi_tri_2))
train_test(X_train, y_train, X_test, y_test)



##### threshold == 3

## unigrams
print("\n===== threshold 3, unigrams =====")
unigrams_3 = list(make_ngram(1, 3, x_train).keys())
vectorized_uni_3 = {key:ind for ind,key in enumerate(unigrams_3)}
X_train = np.array(make_vectorized_data(x_train, unigrams_3, vectorized_uni_3))
clf = svm.SVC(kernel='linear')
print("Training...")
clf.fit(X_train, y_train)
X_test = np.array(make_vectorized_data(x_test, unigrams_3, vectorized_uni_3))
print("Testing...")
y_pred = clf.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}\n")
print(classification_report(y_test, y_pred))

## unigrams + bigrams
print("\n===== threshold 3, unigrams + bigrams =====")
bigrams_3 = list(make_ngram(2, 3, x_train).keys())
uni_bi_3 = unigrams_3 + bigrams_3
vectorized_uni_bi_3 = {key:ind for ind,key in enumerate(uni_bi_3)}
X_train = np.array(make_vectorized_data(x_train, uni_bi_3, vectorized_uni_bi_3))
clf = svm.SVC(kernel='linear')
print("Training...")
clf.fit(X_train, y_train)
X_test = np.array(make_vectorized_data(x_test, uni_bi_3, vectorized_uni_bi_3))
print("Testing...")
y_pred = clf.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}\n")
print(classification_report(y_test, y_pred))

## unigrams + bigrams + trigrams  
print("\n===== threshold 3, unigrams + bigrams + trigrams =====")   
trigrams_3 = list(make_ngram(3, 3, x_train).keys())
uni_bi_tri_3 = uni_bi_3 + trigrams_3
vectorized_uni_bi_tri_3 = {key:ind for ind,key in enumerate(uni_bi_tri_3)}
X_train = np.array(make_vectorized_data(x_train, uni_bi_tri_3, vectorized_uni_bi_tri_3))
clf = svm.SVC(kernel='linear')
print("Training...")
clf.fit(X_train, y_train)
X_test = np.array(make_vectorized_data(x_test, uni_bi_tri_3, vectorized_uni_bi_tri_3))
print("Testing...")
y_pred = clf.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}\n")
print(classification_report(y_test, y_pred))