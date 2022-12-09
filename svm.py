import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


"""
Step 1: read in input args
"""
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True)    # path to training data
parser.add_argument('--test_path', type=str, required=True)     # path to test data 
# parser.add_argument('--output_path', type=str, required=True) # path to output file (metric results)
args = parser.parse_args()

with open(args.train_path, 'r') as train_f:
    col_names = ["text", "label"]
    train_df = pd.read_csv(train_f, delimiter="\t", names=col_names)
    train_df = train_df.dropna()
    train_df = train_df.sample(frac=1, random_state=1)
    x_train = [row[0] for row in train_df.itertuples(index=False)]
    y_train = [row[1] for row in train_df.itertuples(index=False)]

with open(args.test_path, 'r') as test_f:
    col_names = ["text", "label"]
    test_df = pd.read_csv(test_f, delimiter="\t", names=col_names)
    test_df = test_df.dropna()
    test_df = test_df.sample(frac=1, random_state=1)
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
def classify(X_train, y_train, X_test, y_test):
    clf = svm.SVC(kernel='linear')
    print("Training...")
    clf.fit(X_train, y_train)
    print("Testing...")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"Precision: {metrics.precision_score(y_test, y_pred, average='macro')}") 
    print(f"Recall: {metrics.recall_score(y_test, y_pred, average='macro')}") 
    print(f"Macro-f1: {metrics.f1_score(y_test, y_pred, average='macro')}") 
    print(classification_report(y_test, y_pred, digits=4))



"""
Step 3: train & predict
        Linear SVM with default parameters
"""
## unigrams 
# print("\n===== threshold 2, unigrams =====")
print("\n===== unigrams =====")
vectorizer = CountVectorizer(ngram_range=(1,1))
selector = SelectKBest(chi2, k=4000)
X_train = vectorizer.fit_transform(x_train).toarray() # shape: (11417, 44627)
X_train = selector.fit_transform(X_train, y_train)  # shape: (11417, 1000)
X_test = vectorizer.transform(x_test).toarray() # shape: (1164, 44627) # use the fitted selector to transform the test data
X_test = selector.transform(X_test)  # shape: (1164, 1000)                              
classify(X_train, y_train, X_test, y_test)

## unigrams + bigrams
print("\n===== unigrams + bigrams =====")
vectorizer = CountVectorizer(ngram_range=(1,2))
selector = SelectKBest(chi2, k=4000)
X_train = vectorizer.fit_transform(x_train).toarray() # shape: (11417, 44627)
X_train = selector.fit_transform(X_train, y_train)  # shape: (11417, 1000)
X_test = vectorizer.transform(x_test).toarray() # shape: (1164, 44627)
X_test = selector.transform(X_test)  # shape: (1164, 1000)                              
classify(X_train, y_train, X_test, y_test)                            

## unigrams + bigrams + trigrams  
print("\n===== unigrams + bigrams + trigrams =====")   
vectorizer = CountVectorizer(ngram_range=(1,3), max_features=3000, dtype=np.int32)
selector = SelectKBest(chi2, k=3000)
X_train = vectorizer.fit_transform(x_train).toarray() # shape: (11417, 44627)
X_train = np.array(X_train, dtype='int32')
X_train = selector.fit_transform(X_train, y_train)  # shape: (11417, 1000)
X_test = vectorizer.transform(x_test).toarray() # shape: (1164, 44627)
X_test = np.array(X_test, dtype='int32')
X_test = selector.transform(X_test)  # shape: (1164, 1000)                              
classify(X_train, y_train, X_test, y_test)