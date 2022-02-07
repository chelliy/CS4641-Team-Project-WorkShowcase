#!/usr/local/bin/python3
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import time

# Load Dataset
originalData = np.genfromtxt("CollegeData.csv", delimiter=',')
X = originalData[:10000]
y = np.genfromtxt("DraftBoolean.csv", delimiter=',').astype('int32')[:10000]

# Set up random forest model with hyperparameters
randomForest = RandomForestClassifier(n_estimators=10,max_features=1,max_depth=3)

# K fold cross validation
kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(randomForest, X, y, scoring='accuracy', cv=kfold, n_jobs=-1, error_score='raise')
print("Mean Accuracy: " + str(np.mean(n_scores)))
print("Standard Deviation Accuracy: " + str(np.std(n_scores)))

# Predict
randomForest.fit(X, y)
item = np.array(originalData[9994]).reshape(1,-1)
ynew = randomForest.predict(item)
print("Draft Prediction: " + str(ynew[0]))
