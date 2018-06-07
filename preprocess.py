import pandas as pd
import re
import numpy as np

df = pd.read_table('./Delicious/train-label.dat', delimiter=' ', header=None)
df1 = df.sum(axis=0)
maxClass = df1.argmax()
# print(df.iloc[:, maxClass])
df2 = df.iloc[:, maxClass]
print(df2.dtypes)

filepath = './Delicious/train-data.dat'
with open(filepath) as fp:
    line = fp.readline()
    line = re.sub('<.*?>', '', line)
    f = open('./Delicious/trainSet.dat', 'w+')
    while line:
        for x in range(0, len(df2.index)):
            print(str(df2.iloc[x]) + "," + line.strip())
            f.write(line.strip() + "\n")
            line = fp.readline()
            line = re.sub('<.*?>', '', line)
f.close()


df = pd.read_table('./Delicious/test-label.dat', delimiter=' ', header=None)
df1 = df.sum(axis=0)
maxClass = df1.argmax()
# print(df.iloc[:, maxClass])
df2 = df.iloc[:, maxClass]

filepath = './Delicious/test-data.dat'
with open(filepath) as fp:
    line = fp.readline()
    line = re.sub('<.*?>', '', line)
    f = open('./Delicious/testSet.dat', 'w+')
    while line:
        for x in range(0, len(df2.index)):
            print(str(df2.iloc[x]) + " " + line.strip())
            f.write(line.strip() + "\n")
            line = fp.readline()
            line = re.sub('<.*?>', '', line)
f.close()

# infile = filepath
# outfile = "cleaned_file.csv"
#
# fin = open(infile)
# fout = open(outfile, "w+")
# for line in fin:
#     line = re.sub('<.*?>', '', line)
#     fout.write(line)
# fin.close()
# fout.close()
train = pd.read_table("./Delicious/train.dat", encoding='utf_8', header=None)
# data = pd.DataFrame
# for x in range(0, 10):
#     print(train.iloc[x])
# print(train)
# for x in range (8252):
#     for y in range(10):
#         print(x,y)
# vocabs 8520
# docs 8252
matrix = np.zeros((8252, 8520), dtype=int)
matrix1 = np.zeros((3984, 8520), dtype=int)
# for i in range(20):
#     for j in range(8520):
#         if matrix[i][j] > 2:
#             matrix[i][j] = 0
#         print(matrix[i][j], end=' ')
#         c += 1
#     print()

with open('./Delicious/trainSet.dat', 'r') as fp:
    line = fp.readline()
    cnt = 0
    while line:
        for word in line.split():
            a = int(word)
            matrix[cnt][a] = 1
        line = fp.readline()
        cnt += 1

f = open('train.dat', 'w+')
for i in range(8252):
    for j in range(8520):
        if matrix[i][j] == 1:
            f.write(str(1) + "\t")
        else:
            f.write(str(0) + "\t")
    f.write("\n")

with open('./Delicious/testSet.dat', 'r') as fp:
    line = fp.readline()
    cnt = 0
    while line:
        for word in line.split():
            a = int(word)
            matrix1[cnt][a] = 1
        line = fp.readline()
        cnt += 1

f = open('test.dat', 'w+')
for i in range(3984):
    for j in range(8520):
        if matrix1[i][j] == 1:
            f.write(str(1) + "\t")
        else:
            f.write(str(0) + "\t")
    f.write("\n")