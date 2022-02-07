#!/usr/local/bin/python3
FILE = 'CollegeBasketballPlayers2009-2021.csv'
DRAFTED = 'Drafted.txt'
import numpy as np

draftLines = open(DRAFTED, 'r').readlines()
draftLines = [i.strip() for i in draftLines]
draftSet = set(draftLines)

fileLines = open(FILE, 'r').readlines()

filteredLines = []
draftLines = []
for i in fileLines:
    if i.count(',') == 65:
        filteredLines.append(i.strip())
        if i.split(',')[0] in draftSet:
            draftLines.append("1")
        else:
            draftLines.append("0")

draftBooleanFile = open("DraftBoolean.csv", "w")
draftBooleanFile.write(",".join(draftLines))
draftBooleanFile.close()
dataFile = open("CollegeDataTMP.csv", "w")
dataFile.write("\n".join(filteredLines))
dataFile.close()

def cleanDataset(csvFile,savetxt=False,outputName="out.csv"):
    dataset = np.genfromtxt(csvFile, delimiter=',')
    print(dataset.shape)

    filteredDataset = []
    for i in range(0, dataset.shape[1]):
        if not np.isnan(dataset[:,i]).any():
            filteredDataset.append(dataset[:,i])
    filteredDataset = np.transpose(np.array(filteredDataset))

    if savetxt:
        np.savetxt(outputName, filteredDataset, delimiter=",")

    return filteredDataset

cleanDataset("CollegeDataTMP.csv",savetxt=True,outputName="CollegeData.csv")
