import scipy
import numpy as np
import pandas as pd
import os, re, sys
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset,ClassifierChain
from sklearn import tree
from skmultilearn.adapt import MLkNN
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss,f1_score,accuracy_score,zero_one_loss,jaccard_similarity_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import time
from sklearn.pipeline import make_pipeline

#Paths for labels
tlblfile = './Delicious/test-label.dat'
trlblfile='./Delicious/train-label.dat'

trainLabel = np.loadtxt(trlblfile)#load labels for train
testLabel = np.loadtxt(tlblfile)#load label for test
y_train=sp.csr_matrix(trainLabel)
y_test=sp.csr_matrix(testLabel)

#print results
def print_results(clf, label='DEFAULT',Method='DEFAULT'):
	print('********* ' + label+"-"+Method + " *********")
	predictions = clf.predict(X_test)
	print("Accuracy: %f"%(accuracy_score(y_test, predictions)))
	print("hamming_loss: %f"%(hamming_loss(y_test, predictions)))
	print("f1_score_micro: %f"%(f1_score(y_test, predictions, average="micro")))
	print("f1_score_macro: %f"%(f1_score(y_test, predictions, average="macro")))

#fit models
def fit_clf(*args):
        pipeline = make_pipeline(*args)
        pipeline.fit(X_train, y_train)
        return pipeline

#use this to transform train-data.dat ,test-data.dat and pad sentences length
def data(file,path):
	fw = open(path, "w+")
	docs = []
	wordc = 0
	fp = open(file)
	count=0
	while True:
		ln = fp.readline()
		if len(ln) == 0:
			break
		sents = re.findall('<[0-9]*?>([0-9 ]*)', ln)
		for n, i in enumerate(sents):
			words = i.split()
			if (count ==0):
				for w in words:
					wordc += 1
				if n >= 1:
					txt = ' '.join(['%s' % (int(w)) for w in words])
					txt.strip()
					fw.write(txt+" ")
				elif n == 0:
					wordnumber = 279 - wordc
					for i in range(wordnumber):
						fw.write("0 ")
					fw.write("\n")
					wordc = 0
	if(path==pathtest):
		for i in range(260):
			fw.write("0 ")
	elif(path==pathtrain):
		for i in range(253):
			fw.write("0 ")
	fp.close()
	return docs

#paths for read data
trfile='./Delicious/train-data.dat'
tfile='./Delicious/test-data.dat'
#paths for write new data
pathtest="./Delicious/testData.txt"
pathtrain="./Delicious/trainData.txt"

#if used they ruin the last sentences  of each need  extra zeros to reach 280 words per line
data(tfile,pathtest)
data(trfile,pathtrain)

X_train = np.loadtxt(pathtrain)#load new data for train
X_train= np.delete(X_train, (0), axis=0)
X_test=np.loadtxt(pathtest)#load  new data for test
X_test= np.delete(X_test, (0), axis=0)
#print(X_train)
#print(X_test)
#print(X_train.shape)
#print(X_test.shape)

para= [

	{'name': 'Tree', 'obj': tree.DecisionTreeClassifier()},
	{'name': 'Naive bayes', 'obj':MultinomialNB(alpha=0.7)},
	{'name': 'SVM', 'obj': SVC()},
]

for p in para:
	clfs = [
		{'name': 'Classifier Chain', 'obj': ClassifierChain(p['obj'])},
		{'name': 'Binary Relevance', 'obj': BinaryRelevance(classifier =p['obj'], require_dense = [True, True])},
		{'name': 'Label PowerSet', 'obj': LabelPowerset(p['obj'])}
	]
	for c in clfs:
		print_results(fit_clf(c['obj']), c['name'],p['name'])  # print results
