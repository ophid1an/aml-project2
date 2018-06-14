import math

import numpy as np
import scipy.spatial.distance as dist
from scipy.spatial.distance import directed_hausdorff


class CitationKNN(object):

    def __init__(self):
        self._bags = None
        self._bag_predictions = None
        self._labels = None
        self._full_bags = None
        self._DM = None

    def fit(self, train_bags, train_labels, **kwargs):
        self._bags = train_bags
        self._labels = train_labels
        self._no_of_references = kwargs['references']
        self._no_of_citers = kwargs['citers']

    def predict(self, Testbags):
        train_bags = self._bags
        pred_labels = np.array([])
        self._DM = self.DistanceMatrixCKNN(train_bags)

        for i in range(0, len(Testbags)):

            citers = []

            distances = []

            for j in range(0, len(train_bags)):
                distance = directed_hausdorff(Testbags[i].toarray(), train_bags[j].toarray())[0]
                distances.append(distance)
                self._DM[j].append(distance)

            self._DM.append(distances)
            last = len(self._DM) - 1
            self._DM[last].append(0)

            arr = np.array(self._DM[last])
            references = arr.argsort()[:self._no_of_references + 1]

            index = np.argwhere(references == last)
            references = np.delete(references, index)

            for j in range(0, len(self._DM) - 1):
                arr = np.array(self._DM[j])
                neighbors = arr.argsort()[:self._no_of_citers + 1]
                if last in neighbors:
                    citers.append(j)

            relevant_test_labels = []
            for j in range(0, len(references)):
                relevant_test_labels.append(self._labels[references[j]])
            for j in range(0, len(citers)):
                relevant_test_labels.append(self._labels[citers[j]])

            relevant_test_labels.sort()

            label_out = relevant_test_labels[int(math.floor((len(references) + len(citers) - 1) / 2))]
            pred_labels = np.append(pred_labels, label_out)

            self._DM.pop()
            for j in range(0, len(self._DM)):
                self._DM[j].pop()

        return pred_labels

    def DistanceMatrixCKNN(self, full_bag):
        w, h = len(full_bag), len(full_bag)
        Matrix = [[0 for x in range(w)] for y in range(h)]
        for i in range(0, len(full_bag)):
            for j in range(0, len(full_bag)):
                Matrix[i][j] = max(directed_hausdorff(full_bag[i].toarray(), full_bag[j].toarray())[0],
                                   directed_hausdorff(full_bag[j].toarray(), full_bag[i].toarray())[0])

        return Matrix
