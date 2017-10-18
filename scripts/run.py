import numpy as np

from proj1_helpers import *
from implementations import *

TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/test.csv'

if __name__ == "__main__":

	# load training and testing sets
    trainY,trainX,trainIds = load_csv_data(TRAIN_FILE)
    testY,testX,testIds = load_csv_data(TEST_FILE)