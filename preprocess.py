import pandas as pd
import re

df = pd.read_table('./Delicious/train-label.dat', delimiter=' ', header=None)
df1 = df.sum(axis=0)
maxClass = df1.argmax()
# print(df.iloc[:, maxClass])
df2 = df.iloc[:, maxClass]

filepath = './Delicious/train-data.dat'
with open(filepath) as fp:
    line = fp.readline()
    line = re.sub('<.*?>', '', line)
    f = open('./Delicious/train.dat', 'w+')
    while line:
        for x in range(0, len(df2.index)):
            print(str(df2.iloc[x]) + " " + line.strip())
            f.write(str(df2.iloc[x]) + " " + line.strip() + "\n")
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
    f = open('./Delicious/test.dat', 'w+')
    while line:
        for x in range(0, len(df2.index)):
            print(str(df2.iloc[x]) + " " + line.strip())
            f.write(str(df2.iloc[x]) + " " + line.strip() + "\n")
            line = fp.readline()
            line = re.sub('<.*?>', '', line)
f.close()