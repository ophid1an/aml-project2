import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain

RANDOM_STATE = 0
VOCAB_SIZE = 8520
WRITE_RESULTS = False

# Paths for labels files
trainLabelPath = './Delicious/train-label.dat'
testLabelPath = './Delicious/test-label.dat'

# Paths for data files
trainPath = './Delicious/train-data.dat'
testPath = './Delicious/test-data.dat'

# Load labels
y_train = np.loadtxt(trainLabelPath)  # load labels for train
y_test = np.loadtxt(testLabelPath)  # load label for test

# Dataframe to store results
columns = ['accuracy', 'hamming_loss', 'f1_micro', 'f1_macro']
results = pd.DataFrame()


# Data transformation function
def transform_data(source):
    stream = open(source)
    docs = []
    while True:
        ln = stream.readline()
        if len(ln) == 0:
            break
        doc = np.zeros(VOCAB_SIZE, int)
        words = filter(lambda x: x[0] != '<', ln.rstrip('\n').split(' '))
        for w in words:
            doc[int(w)] += 1
        docs.append(doc)

    stream.close()

    docs = np.array(docs)
    return docs


# Predict classifier results function
def predict_results(clf, index):
    predictions = clf.predict(X_test)

    clf_results = [
        accuracy_score(y_test, predictions),
        hamming_loss(y_test, predictions),
        f1_score(y_test, predictions, average='micro'),
        f1_score(y_test, predictions, average='macro'),
    ]

    clf_dataframe = pd.DataFrame(data=[clf_results], index=[index], columns=columns)
    return clf_dataframe


# Plot scores function
def plot_score(results, score):
    ind = np.arange(results.shape[0])
    width = 0.6

    plt.ylabel(score)
    plt.xticks(ind, results.index.values)
    plt.bar(ind, results[score], width)
    plt.show()


# Transform data
X_train = pd.DataFrame(transform_data(trainPath))
X_test = pd.DataFrame(transform_data(testPath))

classifiers = [
    {'name': 'Tree', 'obj': tree.DecisionTreeClassifier(random_state=RANDOM_STATE)},
    {'name': 'NB', 'obj': MultinomialNB(alpha=0.7)},
    {'name': 'SVM', 'obj': SVC(random_state=RANDOM_STATE)},
]

methods = [
    {'name': 'CC', 'obj': lambda clf: ClassifierChain(classifier=clf, require_dense=[False, True])},
    {'name': 'BR', 'obj': lambda clf: BinaryRelevance(classifier=clf, require_dense=[False, True])},
    {'name': 'LP', 'obj': lambda clf: LabelPowerset(classifier=clf, require_dense=[False, True])}
]

for clf in classifiers:
    for method in methods:
        start = time.time()
        # Append classifier results to results
        clf_results = predict_results((method['obj'](clf['obj'])).fit(X_train, y_train),
                                      clf['name'] + '-' + method['name'])
        results = results.append(clf_results)
        # Print classifier results
        print(clf_results)
        stop = time.time()
        print('\nTime: %.3f\n' % (stop - start))

# Print total results
print(results)

# Plot scores
for score in columns:
    plot_score(results, score)

if WRITE_RESULTS:
    filename = 'results-partA.csv'
    results.round(decimals=4).to_csv(filename)
