import numpy as np

from proj1_helpers import *
from implementations import *
from validation import *
from model_enum import *
from feature_preprocessing import *

TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/test.csv'
CHOSEN_MODEL = Model.LEAST_SQUARES
ONLY_SAMPLE = True

if __name__ == "__main__":
    # load training and testing sets
    trainY,trainX,trainIds = load_csv_data(TRAIN_FILE, ONLY_SAMPLE)
    testY,testX,testIds = load_csv_data(TEST_FILE, ONLY_SAMPLE)

    #loss,score = k_fold_cross_validation(trainY, trainX, 2, CHOSEN_MODEL)

    train_processed_x = featuring(trainX)
	test_procedded_x = featuring(testX)

	weight, loss = least_squares(trainY, train_processed_x)
	y_pred = predict_labels(weight, test_procedded_x)

	create_csv_submission(testIds,y_pred, "sumbmission.csv")