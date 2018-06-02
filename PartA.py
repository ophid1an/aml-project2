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
import seaborn as sns
import matplotlib.pyplot as plt

#Paths for labels
tlblfile = './Delicious/test-label.dat'
trlblfile='./Delicious/train-label.dat'

trainLabel = np.loadtxt(trlblfile)#load labels for train
testLabel = np.loadtxt(tlblfile)#load label for test
y_train=sp.csr_matrix(trainLabel)
y_test=sp.csr_matrix(testLabel)

#variable to store results
acc=[]
ham_loss=[]
f1_micro=[]
f1_macro=[]
method=[]
results = pd.DataFrame(columns=['Method', 'acc_score', 'hamming_loss','f1-micro','f1-macro'])
#print results
def print_results(clf, label='DEFAULT',Method='DEFAULT'):
	print('********* ' + label+"-"+Method + " *********")
	predictions = clf.predict(X_test)
	print("Accuracy: %f"%(accuracy_score(y_test, predictions)))
	print("hamming_loss: %f"%(hamming_loss(y_test, predictions)))
	print("f1_score_micro: %f"%(f1_score(y_test, predictions, average="micro")))
	print("f1_score_macro: %f"%(f1_score(y_test, predictions, average="macro")))

	method_name=label+"-"+Method
	method.append(method_name)
	acc.append(accuracy_score(y_test, predictions))
	ham_loss.append(hamming_loss(y_test, predictions))
	f1_micro.append(f1_score(y_test, predictions, average="micro"))
	f1_macro.append(f1_score(y_test, predictions, average="macro"))

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
	{'name': 'Tree', 'obj': tree.DecisionTreeClassifier(random_state=0)},
	{'name': 'NB', 'obj':MultinomialNB(alpha=0.7)},
	{'name': 'SVM', 'obj': SVC(random_state=0)},
]

for p in para:
	clfs = [
		{'name': 'CC', 'obj': ClassifierChain(p['obj'])},
		{'name': 'BR', 'obj': BinaryRelevance(classifier =p['obj'], require_dense = [True, True])},
		{'name': 'LP', 'obj': LabelPowerset(p['obj'])}
	]
	for c in clfs:
		print_results(fit_clf(c['obj']), c['name'],p['name'])  # print results

results["acc_score"]=acc
results["Method"]=method
results["hamming_loss"]=ham_loss
results["f1-micro"]=f1_micro
results["f1-macro"]=f1_macro
print(results)

#Plot accuracy of models
results = results.sort_values(['acc_score'])
colors = sns.color_palette()
ind = np.arange(results.shape[0])
ax = plt.subplot(111)
b = ax.bar(ind - 0.3, results['acc_score'], 0.6, label='acc_score', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.Method)
plt.show()

#plot hamming loss
results = results.sort_values(['hamming_loss'])
colors = sns.color_palette()
ind = np.arange(results.shape[0])
ax = plt.subplot(111)
b = ax.bar(ind - 0.3, results['hamming_loss'], 0.6, label='hamming_loss', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.Method)
plt.show()

#plot f1-micro
results = results.sort_values(['f1-micro'])
colors = sns.color_palette()
ind = np.arange(results.shape[0])
ax = plt.subplot(111)
b = ax.bar(ind - 0.3, results['f1-micro'], 0.6, label='f1-micro', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.Method)
plt.show()

#plot f1-macro
results = results.sort_values(['f1-macro'])
colors = sns.color_palette()
ind = np.arange(results.shape[0])
ax = plt.subplot(111)
b = ax.bar(ind - 0.3, results['f1-micro'], 0.6, label='f1-macro', color=colors[0])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([-0.5, ind[-1] + .5])
ax.set_xticks(ind)
ax.set_xticklabels(results.Method)
plt.show()