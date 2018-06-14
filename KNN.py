import math

import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as dist
from scipy.spatial.distance import directed_hausdorff


class KNN(object):

    def __init__(self):
        self._bags = None
        self._bag_predictions = None
        self._labels = None
        self._full_bags = None
        self._DM = None

    def fit(self, train_bags, train_labels, **kwargs):

        self._bags = train_bags
        self._labels = train_labels
        self._K = kwargs['k']

    def predict(self, Testbags):
        train_bags = self._bags
        pred_labels = np.array([])
        self._DM = self.DistanceMatrix(train_bags, Testbags)

        for i in range(0, len(self._DM)):
            arr = np.array(self._DM[i])
            ind = arr.argsort()[:self._K]
            relevant_test_labels = []
            for j in range(0, len(ind)):
                relevant_test_labels.append(self._labels[ind[j]])
            relevant_test_labels.sort()
            label_out = relevant_test_labels[int(math.floor(self._K / 2))]
            pred_labels = np.append(pred_labels, label_out)
        return pred_labels

    def DistanceMatrix(self, train_bags, test_bags):
        w, h = len(train_bags), len(test_bags)
        Matrix = [[0 for x in range(w)] for y in range(h)]
        for i in range(0, len(test_bags)):
            for j in range(0, len(train_bags)):
                # Matrix[i][j] = _min_hau_bag(test_bags[i], train_bags[j])
                # if i == 0 and j == 0:
                #     print(type(train_bags[i]), type(test_bags[i]))
                Matrix[i][j] = max(directed_hausdorff(test_bags[i].toarray(), train_bags[j].toarray())[0],
                                   directed_hausdorff(train_bags[j].toarray(), test_bags[i].toarray())[0])
        return Matrix

# def _min_hau_bag(X, Y):
#     Hausdorff_distance = max(min((min([list(dist.euclidean(x.toarray(), y.toarray()) for y in Y) for x in X]))),
#                              min((min([list(dist.euclidean(x.toarray(), y.toarray()) for x in X) for y in Y])))
#                              ) if sp.issparse(X) else max(
#         min((min([list(dist.euclidean(x, y) for y in Y) for x in X]))),
#         min((min([list(dist.euclidean(x, y) for x in X) for y in Y])))
#     )
#     return Hausdorff_distance
