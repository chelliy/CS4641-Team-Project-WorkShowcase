#!/usr/local/bin/python3
import numpy as np
import sklearn.metrics

def cleanDataset(csvFile,savetxt=False,outputName="out.csv"):
    dataset = np.genfromtxt(csvFile, delimiter=',')

    filteredDataset = []
    for i in range(0, dataset.shape[1]):
        if not np.isnan(dataset[:,i]).any():
            filteredDataset.append(dataset[:,i])
    filteredDataset = np.transpose(np.array(filteredDataset))

    if savetxt:
        np.savetxt(outputName, filteredDataset, delimiter=",")

    return filteredDataset

def linear_fit_closed(xtrain, ytrain):
    inner = np.matmul(np.transpose(xtrain), xtrain)
    inverse = np.linalg.pinv(inner)
    pseudoInverse = np.matmul(inverse,np.transpose(xtrain))
    theta = np.matmul(pseudoInverse, ytrain).reshape(xtrain.shape[1],1)
    return theta

def ridge_fit_closed(xtrain, ytrain, c_lambda):  # [5pts]
    transposeMult = np.matmul(np.transpose(xtrain), xtrain)
    lambdaMatrix = np.identity(transposeMult.shape[0])*c_lambda
    inverse = np.linalg.pinv(transposeMult + lambdaMatrix)
    pseudoinverse = np.matmul(inverse,np.transpose(xtrain))
    return np.matmul(pseudoinverse, ytrain)

def rmse(pred, label):
    return np.sqrt(np.mean((pred-label)**2))

def predict(xtest, weight):
    return np.sum(xtest*(weight.reshape(weight.shape[0])),axis=-1).reshape(xtest.shape[0],1)

def pcaReduce(data, K=2):
    U,S,V = np.linalg.svd(data, full_matrices=False)
    return np.matmul(data, np.transpose(V))[:,:K]

datasetNBA = np.genfromtxt('finalNBA.csv', delimiter=',')
x = pcaReduce(datasetNBA[:,1:],2)
y = datasetNBA[:,0]
weights = linear_fit_closed(x[0:500,:],y[0:500])
#weights = ridge_fit_closed(x[0:500,:],y[0:500],1.5)
predictions = predict(x[500:,:],weights)
error = sklearn.metrics.r2_score(y[500:],predictions)
print(error)
