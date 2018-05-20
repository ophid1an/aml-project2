import scipy
import numpy as np
import pandas as pd
import os, re, sys
from scipy.io import arff
import scipy.sparse as sp
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.adapt import MLkNN
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss,f1_score,accuracy_score

tlblfile = './Delicious/test-label.dat'
trlblfile='./Delicious/train-label.dat'

trainLabel = np.loadtxt(trlblfile)#load labels for train
testLabel = np.loadtxt(tlblfile)#load label for test
y_train=sp.csr_matrix(trainLabel)
y_test=sp.csr_matrix(testLabel)

#use this to transform train-data.dat ,test-data.dat
#It read the files train-data.dat ,test-data.dat and write to another file
#Dont know if this is correct!!!!!!
def data(file,path):
	fw = open(path, "w+")
	docs = []
	row = []
	wordc = 0
	temp = 0
	fp = open(file)
	count=0
	while True:
		sents = []
		sd = []
		ln = fp.readline()
		if len(ln) == 0:
			break
		Sd = re.findall('^<([0-9]*)>', ln)
		sents = re.findall('<[0-9]*?>([0-9 ]*)', ln)
		for n, i in enumerate(sents):
			words = i.split()
			for w in words:
				wordc += 1
			# print(words)
			if (count > 0):
				if n >= 1:
					# row.append(' '.join(['%d' % (int(w)) for w in words]))
					txt = ' '.join(['%d' % (int(w)) for w in words])
					# print(wordc)
					fw.write(txt + " ")
				elif n == 0:
					# docs.append(row)
					wordnumber = 280 - wordc
					for i in range(wordnumber):
						fw.write("0 ")
					asd = wordc + wordnumber
					fw.write("\n")
					# print(docs)
					wordc = 0
					row = []
			count += 1
	fp.close()
pathtest="./Delicious/testData.txt"
pathtrain="./Delicious/trainData.txt"
trfile='./Delicious/train-data.dat'
tfile='./Delicious/test-data.dat'
#if used they ruin the last sentences  of each need  extra zeros to reach 280 words per line
#data(tfile,pathtest)
#data(trfile,pathtrain)
X_train = np.loadtxt("./Delicious/trainData.txt")#load data for train
X_test = np.loadtxt("./Delicious/testData.txt")#load data for test


#Create dummy data(not used)
#X, y = make_multilabel_classification(sparse = True, n_labels =5,
  #return_indicator = 'sparse', allow_unlabeled = False)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15) #dummy test data



classifier = BinaryRelevance(classifier = SVC(), require_dense = [False, True])
# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)
print("---------BinaryRelevance---------")
print("Accuracy:")
print(accuracy_score(y_test,predictions))
print("hamming_loss:")
print(hamming_loss(y_test, predictions))#lowest the better
print("f1_score")
print(f1_score(y_test, predictions,average="micro"))
