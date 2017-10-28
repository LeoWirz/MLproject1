import numpy as np
import time

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
    train_y,train_x,train_ids = load_csv_data(TRAIN_FILE, ONLY_SAMPLE)
    test_y,test_x,test_ids = load_csv_data(TEST_FILE, ONLY_SAMPLE)

    train_processed_x = featuring(train_x)
    test_processed_x = featuring(test_x)

    lambdas = np.logspace(-4, 0, 10)
    rmse_tr_list = []
    rmse_te_list = []
    for l in lambdas:
        rmse_tr, rmse_te, score = k_fold_cross_validation(train_y,
                                                          train_processed_x,
                                                          4,
                                                          CHOSEN_MODEL,
                                                          lambda_ = l)
        rmse_tr_list.append(rmse_tr)
        rmse_te_list.append(rmse_te)
    best_lambda = lambdas[np.argmin(rmse_te_list)]

    weight, loss = launch_model_function(train_y, train_processed_x, CHOSEN_MODEL, lambda_ = best_lambda)
    y_pred = predict_labels(weight, test_processed_x)

    timestr = time.strftime("%Y%m%d%H%M%S")

    create_csv_submission(test_ids,y_pred, "submission{}.csv".format(timestr))