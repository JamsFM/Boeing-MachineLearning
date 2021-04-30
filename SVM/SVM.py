import os

import numpy as np #TODO: Check if these libraries are needed
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import random

#for labels (not being used atm, used with file that has an array)
import json

def ProcessLogs(): 

    from tkinter import Tk     # from tkinter import Tk for Python 3.x
    from tkinter.filedialog import askopenfilename


    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    fileNames = []
    for i in range(5): #in order: trainingSet,testingSet,trainingLabels,testingLabels,vectorizerModel
        fileNames.append(askopenfilename())

    global trainingLogs, testingLogs, trainingLabels, testingLabels
    with open(fileNames[4], 'rb') as f:
        vectorizer = pickle.load(f)

    with open(fileNames[0], 'r', encoding='utf8') as f:
        vectors = vectorizer.transform(f)
        dense = vectors.todense()
        trainingLogs = np.asarray(dense)

    with open(fileNames[1], 'r', encoding='utf8') as f:
        vectors = vectorizer.transform(f)
        dense = vectors.todense()
        testingLogs = np.asarray(dense)

    with open(fileNames[2], 'r', encoding='utf8') as f:
        trainingLabels = json.load(f)


    with open(fileNames[3], 'r', encoding='utf8') as f:
        testingLabels = json.load(f)

    labelTranslation = ["safe", "ssh", "ws", "sql", "ddos", "ps", "su"]
    trainingLabelsString = []
    testingLabelsString = []
    for i in range(len(trainingLabels)):
        label = trainingLabels[i]
        trainingLabels[i] = labelTranslation[label]
    for i in range(len(testingLabels)):
        label = testingLabels[i]
        testingLabels[i] = labelTranslation[label]
    

    # Returns position in file to original file
    testingSet = open(fileNames[1], 'r', encoding="utf8")
    global logs 
    logs = []
    for message in testingSet:
        logs.append(message)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def ModelTrain(): # Train and save a support vector machines (SVM) algorithm
    

    clf = SVC(kernel='rbf') # Uses a radial basis function kernel
    clf.fit(trainingLogs, trainingLabels)
    predictions = clf.predict(trainingLogs)
    print("Accuracy score of trained sklearn model: " + str(accuracy_score(trainingLabels,predictions)))
    pickle.dump(clf, open('SVMModel.sav', 'wb')) # Save the model

def MakePredictions(): # Writes each log and its corresponding prediction to a file called "AlgorithmOutput.txt"
    # Loads model, and makes prediction
    loadedModel = pickle.load(open('SVMModel.sav', 'rb'))
    predictions = loadedModel.predict(testingLogs)
    result = loadedModel.score(testingLogs, testingLabels)
    print("Loaded model accuracy: " + str(result))
    print(predictions)


    #write predictions to output file
    f = open("AlgorithmOutput.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "AlgorithmOutput.txt" if one doesn't already exist
    frequencies = {'safe': 0, 'ws': 0, 'sql': 0, 'ddos': 0, 'ps': 0, 'ssh': 0, 'su': 0}
    for i in range(len(predictions)):
        frequencies[predictions[i]] = (frequencies.get(predictions[i]) + 1)
    for badActor in frequencies:
        frequencies[badActor] = (frequencies.get(badActor)/len(predictions))*100
   
    for i in range(len(predictions)): # Write every log and its corresponding prediction to file
        f.write("Prediction: ")
        f.write(predictions[i]) # Write prediction to file
        f.write(" Frequency: ")
        f.write(f'{frequencies.get(predictions[i]):.4g}')
        f.write("%   log: ")
        f.write(logs[i]) # Write original log to file (not inluding '\n')
        #f.write("\n")
    f.close()

ProcessLogs()
ModelTrain()
MakePredictions() 
