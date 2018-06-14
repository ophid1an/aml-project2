import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.models import *
from libact.query_strategies import *
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0
VOCAB_SIZE = 8520
TEST_SIZE = 0.5
INITIAL_TRAIN_SIZE = 10
SAMPLES_NUM = 10

testPath = './Delicious/test-data.dat'
testLabelPath = './Delicious/test-label.dat'


# Data transformation function
def transform_data(source):
    with open(source) as fp:
        line = fp.readline()
        docs = []
        while line:
            doc = np.zeros(VOCAB_SIZE, int)
            words = filter(lambda x: x[0] != '<', line.rstrip('\n').split(' '))
            for w in words:
                doc[int(w)] += 1
            docs.append(sp.csr_matrix(doc))
            line = fp.readline()

    docs = np.array(docs)
    return docs


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test(X, y, test_size, n_labeled):
    X_, y_ = Dataset(X, y).format_sklearn()

    X_train, X_test, y_train, y_ = \
        train_test_split(X_, y, test_size=test_size)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


# Transform dataset
X = transform_data(testPath)

y = pd.read_table(testLabelPath, delimiter=' ', header=None)
least_common_class = y.sum(axis=0).values.argmin()
y = y.iloc[:, least_common_class]

# Load dataset
trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
    split_train_test(X, y, TEST_SIZE, INITIAL_TRAIN_SIZE)
trn_ds2 = copy.deepcopy(trn_ds)
lbr = IdealLabeler(fully_labeled_trn_ds)

# Comparing UncertaintySampling strategy with RandomSampling.
qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
model = LogisticRegression()
E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, SAMPLES_NUM)

qs2 = RandomSampling(trn_ds2)
model = LogisticRegression()
E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, SAMPLES_NUM)

# Plot the learning curve of UncertaintySampling to RandomSampling
# The x-axis is the number of queries, and the y-axis is the corresponding
# error rate.
query_num = np.arange(1, SAMPLES_NUM + 1)
plt.plot(query_num, E_in_1, 'b', label='qs Ein')
plt.plot(query_num, E_in_2, 'r', label='random Ein')
plt.plot(query_num, E_out_1, 'g', label='qs Eout')
plt.plot(query_num, E_out_2, 'k', label='random Eout')
plt.xlabel('Number of Queries')
plt.ylabel('Error')
plt.title('Experiment Result')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.show()
