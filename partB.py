import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# TODO: Parse correctly
# train = pa.read_table("./Delicious/train-data.dat", delimiter=' ', header=None)
# test = pa.read_table("./Delicious/test-data.dat", delimiter=' ', header=None)

x_train = pd.read_table("train.dat", encoding='utf_8', header=None)
x_test = pd.read_table("test.dat", encoding='utf_8', header=None)

x_train = x_train.replace(np.nan, 0)
x_test = x_test.replace(np.nan, 0)

df = pd.read_table('./Delicious/train-label.dat', delimiter=' ', header=None)
df2 = pd.read_table('./Delicious/test-label.dat', delimiter=' ', header=None)
df1 = df.sum(axis=0)
maxClass = df1.argmax()

y_train = df.iloc[:, maxClass]
y_test = df2.iloc[:, maxClass]

train_bags = x_train[:-1]
train_labels = y_train
test_bags = x_test[:-1]
test_labels = y_test

# from sklearn.ensemble import RandomForestRegressor
#
# rf = RandomForestRegressor(n_estimators=10, random_state=42)
# rf.fit(train_bags, train_labels)
# predictions = rf.predict(test_bags)
# errors = abs(predictions - test_labels)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# mape = 100 * (errors / test_labels)
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
#
# dt = DecisionTreeClassifier()
# clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
# clf.fit(x_train[:-1],y_train)

# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
# text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
# twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# predicted = text_clf.predict(twenty_test.data)
# np.mean(predicted == twenty_test.target)

# from sklearn.linear_model import SGDClassifier
# text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
# predicted_svm = text_clf_svm.predict(twenty_test.data)
# np.mean(predicted_svm == twenty_test.target)