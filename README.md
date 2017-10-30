# Machin learning project 1
Team : Boris, Matteo, Leonardo

In order to launch the analysis, go the the scripts folder and run the run.py script. 

The run will first import the train and test data stored in the folder data, and clean the data. Make sure that both files "train.csv" and "text.csv" are in the data folder, as they are to large to be on the github repository.
An important part of the cleaning process is the split of the dataset into 4 subset, according to the categorical feature 0,1,2 and 3 from the feature 22.

Then, it will launch a logistic regression for each of the 4 subsets, with 10'000 iterations each, and a gamma of 10^-7.
Remark: Do to the fact that we run 4 separate logistic regression, 5 minutes for each, it takes +/- 20 minutes to compute the weights.

The output file containing the prediction will be created in the script folder.