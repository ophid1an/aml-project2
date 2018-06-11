import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import tree
from sklearn.metrics import hamming_loss, f1_score, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain

RANDOM_STATE = 0

# Paths for labels files
trainLabelPath = './Delicious/train-label.dat'
testLabelPath = './Delicious/test-label.dat'

# Paths for data files
trainPath = './Delicious/train-data.dat'
testPath = './Delicious/test-data.dat'

trainLabel = np.loadtxt(trainLabelPath)  # load labels for train
testLabel = np.loadtxt(testLabelPath)  # load label for test
y_train = sp.csr_matrix(trainLabel)
y_test = sp.csr_matrix(testLabel)

# Dataframe to store results
columns = ['accuracy', 'hamming_loss', 'f1_micro', 'f1_macro']
results = pd.DataFrame()


# Data transformation function
# use this to transform train-data.dat ,test-data.dat and pad sentences length
def data(file, path):
    fw = open(path, "w+")
    docs = []
    wordc = 0
    fp = open(file)
    count = 0
    while True:
        ln = fp.readline()
        if len(ln) == 0:
            break
        sents = re.findall('<[0-9]*?>([0-9 ]*)', ln)
        for n, i in enumerate(sents):
            words = i.split()
            if count == 0:
                wordc += len(words)
                if n >= 1:
                    txt = ' '.join(['%s' % (int(w)) for w in words])
                    txt.strip()
                    fw.write(txt + " ")
                elif n == 0:
                    wordnumber = 279 - wordc
                    for i in range(wordnumber):
                        fw.write("0 ")
                    fw.write("\n")
                    wordc = 0
    if path == pathtest:
        for i in range(260):
            fw.write("0 ")
    elif path == pathtrain:
        for i in range(253):
            fw.write("0 ")
    fp.close()
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

    plt.title(score)
    plt.xticks(ind, results.index.values)
    plt.bar(ind, results[score], width)
    plt.show()


# paths for write new data
pathtest = "./Delicious/testData.txt"
pathtrain = "./Delicious/trainData.txt"

# if used they ruin the last sentences  of each need  extra zeros to reach 280 words per line
data(testPath, pathtest)
data(trainPath, pathtrain)

X_train = np.loadtxt(pathtrain)  # load new data for train
X_train = np.delete(X_train, 0, axis=0)
X_test = np.loadtxt(pathtest)  # load  new data for test
X_test = np.delete(X_test, 0, axis=0)

classifiers = [
    {'name': 'Tree', 'obj': tree.DecisionTreeClassifier(random_state=RANDOM_STATE)},
    {'name': 'NB', 'obj': MultinomialNB(alpha=0.7)},
    {'name': 'SVM', 'obj': SVC(random_state=RANDOM_STATE)},
]

methods = [
    {'name': 'CC', 'obj': lambda p: ClassifierChain(p)},
    {'name': 'BR', 'obj': lambda p: BinaryRelevance(classifier=p, require_dense=[True, True])},
    {'name': 'LP', 'obj': lambda p: LabelPowerset(p)}
]

for clf in classifiers:
    for method in methods:
        start = time.time()
        # Append classifier results to results
        clf_results = predict_results((method['obj'](clf['obj'])).fit(X_train, y_train),
                                      clf['name'] + ' - ' + method['name'])
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
