import pandas as pd
import re

# TODO: Parse correctly
# train = pa.read_table("./Delicious/train-data.dat", delimiter=' ', header=None)
# test = pa.read_table("./Delicious/test-data.dat", delimiter=' ', header=None)

df = pd.read_table('./Delicious/train-label.dat', delimiter=' ', header=None)
df1 = df.sum(axis=0)
maxClass = df1.argmax()
# print(df.iloc[:, maxClass])
df2 = df.iloc[:, maxClass]
print(df2)
for x in range(0, len(df2.index)):
    print(df2.iloc[x])
# train = pd.read_table('./Delicious/train.dat', delimiter=' ', header=None)
# train.info()

# train = pd.concat([df2, train], axis=1)
# train.info()


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
