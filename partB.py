import pandas as pa

# TODO: Parse correctly
train = pa.read_csv("./Delicious/train-data.dat", delimiter=' ', header=None)
test = pa.read_csv("./Delicious/test-data.dat", delimiter=' ', header=None)
