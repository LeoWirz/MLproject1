import numpy as np
from proj1_helpers import *
from implementations import *

# ---------- Import the data ---------------

DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

# Load the training data into feature matrix, class labels, and event ids:
print("Load the Data")
y, X, ids = load_csv_data(DATA_TRAIN_PATH)
ytest, Xtest, idstest = load_csv_data(DATA_TEST_PATH)

# Convert labels in order to use the logistic regression
y[y==-1] = 0

# ----------- Clean the Data ---------------
print("Clean the data")

# Create the 4 models
models = split_by_label(X, y, ids, [0,1,2,3], 22)
models_test = split_by_label(Xtest, ytest, idstest, [0,1,2,3], 22)

# Clean the 4 models
for m in models:
    m[0] = clean(m[0])
    
for m in models_test:
    m[0] = clean(m[0])

# Standardize the matrix of experience
for m in models:
    m[0] = standardize_m(m[0])
    
for m in models_test:
    m[0] = standardize_m(m[0])

# ------------ Compute the weights and prepare the testing data --------------
print("Compute logistic")
lambda_ = 0.0000001
iterations = 10000
degree = 1

W, losses = magic(models, iterations, lambda_, degree)

# Create test model of right size
print("Prepare the test data")
for ind, m in enumerate(models_test):
    
    if ind is 0:
        m[0] = build_comb_poly_log_sqrt_m(m[0], degree) 
        
    if ind is 1:
        m[0] = build_comb_poly_log_sqrt_m(m[0], degree)
        
    if ind is 2:
        m[0] = build_comb_poly_log_sqrt_m(m[0], degree) 
        
    if ind is 3:
        m[0] = build_comb_poly_log_sqrt_m(m[0], degree)

# ----------------- Create the predictions ----------------------------

# Find predictions
print("Find Predictions")

Y_preds = []

for i in range(len(models_test)):
    y = predict_labels(W[i], models_test[i][0])
    Y_preds.append(y)

# Create one vector of prediction, sorted by IDS
Y_pred = np.concatenate((Y_preds[0], Y_preds[1], Y_preds[2], Y_preds[3]), axis=0)
ids_pred = np.concatenate((models_test[0][2], models_test[1][2], models_test[2][2], models_test[3][2]), axis=0)

pred = np.column_stack((ids_pred, Y_pred))
# Sort by IDS
pred = pred[np.argsort(pred[:, 0])]

# Create submission file
print("Create submission file")
create_csv_submission(pred[:,0], pred[:,1], "predictions.csv")
