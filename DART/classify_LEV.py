import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

parser = argparse.ArgumentParser()
parser.add_argument('--clean_lev_path', type=str)       # input path to LEV data      (e.g. "clean_LEV.tsv")
parser.add_argument('--clean_nonlev_path', type=str)    # input path to non-LEV data  (e.g. "clean_NONLEV.tsv") 
parser.add_argument('--osact_train_path', type=str)           # input path to OSACT train data (e.g. ../OSACT/osact_train_cleaned.tsv)
parser.add_argument('--osact_lev_output_path', type=str)      # output path to OSACT train data that got classified as Levantine
args = parser.parse_args()

"""
Create train-test splits of Levantine & non-Levantine data.
"""
with open(args.clean_lev_path, 'r') as f1:
    lev = pd.read_csv(f1, delimiter="\t", header=None, index_col=False)
with open(args.clean_nonlev_path, 'r') as f2:
    nonlev = pd.read_csv(f2, delimiter="\t", header=None, index_col=False)

data = pd.concat([lev, nonlev], axis=0, ignore_index=True)
train_df, test_df = train_test_split(data, shuffle=True, test_size=0.2)  # shuffle & use 80train/20test
train_df = train_df.dropna()
test_df = test_df.dropna()

### Create train and test files
train_df.to_csv('classify_lev_train.tsv', sep="\t", index=False, header=False)
test_df.to_csv('classify_lev_test.tsv', sep="\t", index=False, header=False)

x_train = [row[0] for row in train_df.itertuples(index=False)]
y_train = [row[1] for row in train_df.itertuples(index=False)]
x_test = [row[0] for row in test_df.itertuples(index=False)]
y_test = [row[1] for row in test_df.itertuples(index=False)]


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
    return clf

"""
Step 3: train & predict
        Linear SVM with default parameters
"""
## unigrams 
vectorizer = TfidfVectorizer()
selector = SelectKBest(chi2, k=500)
X_train = vectorizer.fit_transform(x_train).toarray()
X_train = selector.fit_transform(X_train, y_train)
X_test = vectorizer.transform(x_test).toarray() # use the fitted selector to transform the test data
X_test = selector.transform(X_test)                              
model = classify(X_train, y_train, X_test, y_test)

# # save the model
MODEL_PATH = 'levantine_classifier.pickle'
pickle.dump(model, open(MODEL_PATH, 'wb'))



# load the OSACT train data
with open(args.osact_train_path, 'r') as f:
    osact_df = pd.read_csv(f, delimiter="\t", header=None, index_col=False, error_bad_lines=False)
    osact_df = osact_df.dropna()

osact_tweets = osact_df.iloc[:, 0].tolist()
osact_labels = osact_df.iloc[:, 1].tolist()
osact_tweets_encoded = vectorizer.transform(osact_tweets).toarray() # use the fitted selector to transform the test data
osact_tweets_encoded = selector.transform(osact_tweets_encoded)

# load the model & test on 
model = pickle.load(open(MODEL_PATH, 'rb'))
predictions = model.predict(osact_tweets_encoded).tolist()

with open(args.osact_lev_output_path, 'w') as f: 
    for ind,p in enumerate(predictions):
        if p=="LEV":
            f.write(f"{osact_tweets[ind]}\t{osact_labels[ind]}\n")