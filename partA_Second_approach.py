import scipy
import numpy as np
import pandas as pd
import os, re, sys
from scipy.io import arff
import scipy.sparse as sp
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss,f1_score,accuracy_score,zero_one_loss,jaccard_similarity_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from itertools import islice
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
vocab='./Delicious/vocabs.txt'
tlblfile = './Delicious/test-label.dat'
trlblfile='./Delicious/train-label.dat'

trainLabel = np.loadtxt(trlblfile)#load labels for train
testLabel = np.loadtxt(tlblfile)#load label for test
vocab= pd.read_csv(vocab, sep=",",names = ["Words", "Number"])#load vocab
vocab.values

#y_train=sp.csr_matrix(trainLabel)
#y_test=sp.csr_matrix(testLabel)

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
			# print(words)
			for w in words:
				nword=str(vocab.Words[int(w)])
			#count += 1
			if (count ==0):
				#for w in words:
					# print(vocab.Words[int(w)])
					#txt = str(vocab.Words[int(w)])
					#fw.write(txt + " ")
					#wordc += 1
				if n >= 1:
					doc=0
					#word=(vocab.Words[int(w)] for w in words)
					#row.append(' '.join(['%s' % (str(vocab.Words[int(w)])) for w in words]))
					txt = ' '.join(['%s' % (str(vocab.Words[int(w)])) for w in words])
					txt.strip()

					fw.write(txt+" ")
					#print(row)
					# print(wordc)
				elif n == 0:
					#docs.append(row)
					wordnumber = 280 - wordc
					#for i in range(wordnumber):
						#fw.write("0 ")
					asd = wordc + wordnumber

					fw.write("\n")
					# print(docs)
					wordc = 0
					row = []
	return docs

	fp.close()
	#return docs

pathtest="./Delicious/testData1.txt"
pathtrain="./Delicious/trainData2.txt"
pathcomplete="./Delicious/completeData.txt"
pathcomplete2="./Delicious/completeData2.txt"
trfile='./Delicious/train-data.dat'
tfile='./Delicious/test-data.dat'

#if used they ruin the last sentences  of each need  extra zeros to reach 280 words per line
#docs=data(pathcomplete,pathcomplete2)
#print(len(docs))
#docs2=data(trfile,pathtrain)
#X=MultiLabelBinarizer().fit_transform(docs)
#X_train=MultiLabelBinarizer().fit_transform(docs2)
#X_train=sp.csr_matrix(X_train)
#X_test=sp.csr_matrix(X_test)
#print(docs)
#vocab=vocab.Number.values
docs= pd.read_csv(pathcomplete2,header=None)
docs.values

docs.info()
#docs = np.delete(docs, (0), axis=0)
docs1= pd.DataFrame(docs,columns=["txt"])
print(docs)
tvec = TfidfVectorizer(vocabulary=vocab.Words,min_df=.0025, max_df=.1, ngram_range=(1, 2),lowercase=False)  # initialize TFIDF VECTORIZER
X= tvec.fit_transform(docs[0])
#transformer = TfidfTransformer()
#X= transformer.fit_transform(X)
X= preprocessing.normalize(X, norm='l2')

Label = np.loadtxt('./Delicious/completeLabels.txt')#load label for test
y=sp.csr_matrix(Label)


print(X)
print(y)
print(X.shape)
print(y.shape)
#X_train = np.loadtxt("./Delicious/trainData2.txt")#load data for train
#X_test = np.loadtxt("./Delicious/testData1.txt")#load data for test
#X_train=sp.csr_matrix(X_train)
#X_test=sp.csr_matrix(X_test)
#print(X_train)
#xtrain="./Delicious/train-data.dat"

#print(X_train)
#Create dummy data(not used)
#X, y = make_multilabel_classification(sparse = True, n_labels =5,
  #return_indicator = 'sparse', allow_unlabeled = False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33) #dummy test data

#One vs Rest Method
#classif = OneVsRestClassifier(SVC(kernel='linear'))
#classif.fit(X_train, y_train)
#predictions = classif.predict(X_test)

#Binary Relevance Method
classifier = BinaryRelevance(classifier =SVC(), require_dense = [False, True])
# train
classifier.fit(X_train, y_train)
# predict
predictions = classifier.predict(X_test)

#MLK method
#classifier = MLkNN(k=3)
# train
#classifier.fit(X_train, y_train)
# predict
#redictions = classifier.predict(X_test)

print(predictions)
print("---------BinaryRelevance---------")
print("Accuracy:")
print(accuracy_score(y_test,predictions))
print("hamming_loss:")
print(hamming_loss(y_test, predictions))#lowest the better
print("Jacard similarity score:")
print(jaccard_similarity_score(y_test, predictions))
print("Zero one loss:")
print(zero_one_loss(y_test, predictions))
#print("f1_macro")
#print(f1_score(y_test, predictions,average='samples'))
