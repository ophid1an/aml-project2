import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from CitationKNN import CitationKNN

RANDOM_STATE = 0
VOCAB_SIZE = 8520
TRAIN_SAMPLES = 100
TEST_SAMPLES = 100

# Paths for labels files
trainLabelPath = './Delicious/train-label.dat'
testLabelPath = './Delicious/test-label.dat'

# Paths for data files
trainPath = './Delicious/train-data.dat'
testPath = './Delicious/test-data.dat'


def transform_data(filepath):
    with open(filepath) as fp:
        line = fp.readline()
        docs = []
        while line:
            line_array = line.split(' ')[1:]

            indices = []  # Indices of the first word of each sentence
            # Populate indices
            for ind, elem in enumerate(line_array):
                if elem[0] == '<':
                    indices.append(ind + 1)

            indices_len = len(indices)

            sentences = []
            # Populate sentences
            for i in range(indices_len):
                if i == indices_len - 1:
                    sentences.append(line_array[indices[i]:])
                else:
                    sentences.append(line_array[indices[i]:indices[i + 1] - 1])

            bag = np.zeros([indices_len, VOCAB_SIZE], int)
            # Populate bag
            for i in range(len(sentences)):
                for elem in sentences[i]:
                    bag[i][int(elem)] += 1

            docs.append(sp.csr_matrix(bag))
            line = fp.readline()

    return docs


X_train = transform_data(trainPath)
X_test = transform_data(testPath)

y_train = pd.read_table(trainLabelPath, delimiter=' ', header=None)
most_common_class = y_train.sum(axis=0).values.argmax()

y_train = y_train.iloc[:, most_common_class]
y_test = pd.read_table(testLabelPath, delimiter=' ', header=None).iloc[:, most_common_class]

# Resampling
X_train, y_train = resample(X_train, y_train, replace=False, n_samples=TRAIN_SAMPLES, random_state=RANDOM_STATE)
X_test, y_test = resample(X_test, y_test, replace=False, n_samples=TEST_SAMPLES, random_state=RANDOM_STATE)

# Apply CKNN
references = [1, 3, 5, 7, 9]
citers = [1, 2, 3, 4, 5]

print('Citation-kNN')
print('************')
for c in citers:
    print('\nUsing %d citer(s)' % c)
    for r in references:
        print('\n\tUsing %d reference(s)' % r)
        start = time.time()
        clf = CitationKNN()
        clf.fit(X_train, np.array(y_train), references=r, citers=c)
        y_pred = clf.predict(X_test)
        stop = time.time()
        print('\tAccuracy: %.3f' % (accuracy_score(y_test, y_pred)))
        print('\tTime: %.3f' % (stop - start))
