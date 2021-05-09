import os

import numpy as np #TODO: Check if these libraries are needed
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
import random

# For labels (not being used atm, used with file that has an array)
import json

# For opening file explorer windows
from tkinter import Tk     # From tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

def ProcessLogs(filesToLoad = "all", cardinality = "multinomial", includeSafeLogs = True, loadNewVectorizer = False): 
    #filesToLoad options: "all", "training", "testing" (logs and labels), "testingLogs"
    Tk().withdraw() # We don't want a full GUI, so keep the root window from appearing
    fileNames = ["","","","",""]
    if filesToLoad == "training" or filesToLoad == "all":
        fileNames[0] = askopenfilename(title= 'Training Set')
        fileNames[1] = askopenfilename(title= (cardinality + ' Training Labels'))
    if filesToLoad == "testing" or filesToLoad == "testingLogs" or filesToLoad == "all":
        fileNames[2] = askopenfilename(title= 'Testing Set')
    if filesToLoad == "testing" or filesToLoad == "all":
        fileNames[3] = askopenfilename(title= (cardinality + ' Testing Labels'))
    if loadNewVectorizer:
        fileNames[4] = askopenfilename(title='vectorizerModel')

    global trainingLogs, testingLogs, trainingLabels, testingLabels
    global logs # logs saves the original logs unprocessed so they can be output later

    if not loadNewVectorizer: #TODO: Figure out how to save the vectorizer model so it doesn't have to be loaded every time.
        #vectorizer = pickle.load(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), 'universalVectorizer.pickle'), 'r') #Opens logs.txt (has to be in same directory as python script)
        #with open(fileNames[4], 'rb') as f: # Load vectorizer
            vectorizer = pickle.load(f)
    else:
        with open(fileNames[4], 'rb') as f: # Load vectorizer
            vectorizer = pickle.load(f)

    if filesToLoad == "training" or filesToLoad == "all":
        with open(fileNames[0], 'r', encoding='utf8') as f: #
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            trainingLogs = np.asarray(dense)

    if filesToLoad == "training" or filesToLoad == "all":
        with open(fileNames[1], 'r', encoding='utf8') as f:
            trainingLabels = json.load(f)
            #trainingLabels = np.asarray(trainingLabels) #TODO: TEST IF WORKS

    if filesToLoad == "testing" or filesToLoad == "testingLogs" or filesToLoad == "all":
        with open(fileNames[2], 'r', encoding='utf8') as f:
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            testingLogs = np.asarray(dense)

    if filesToLoad == "testing" or filesToLoad == "all":
        with open(fileNames[3], 'r', encoding='utf8') as f:
            testingLabels = json.load(f)
            #testingLabels = np.asarray(testingLabels) #TODO: TEST IF WORKS

    if cardinality == "multinomial":
        labelTranslation = ["safe", "ssh", "ws", "sql", "ddos", "ps", "su","unsafe"]
    elif cardinality == "binomial":
        labelTranslation = ["safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe"]
    else:
        print("Entered invalid cardinality to process logs function. Enter \"binomial\" or \"multinomial\".")
        return
    if filesToLoad == "training" or filesToLoad == "all":
        for i in range(len(trainingLabels)): # Change the numeric representation of the bad actors to the string representation.
            trainingLabels[i] = labelTranslation[trainingLabels[i]]
    if filesToLoad == "testing" or filesToLoad == "all":
        for i in range(len(testingLabels)):
            testingLabels[i] = labelTranslation[testingLabels[i]]
    
    if not includeSafeLogs: # Remove all safe logs from training and testing sets
        if filesToLoad == "training" or filesToLoad == "all":
            trainingLogsBadActor = [] # Remove safe logs from training logs
            for i in range(len(trainingLogs)):
                if trainingLabels[i] != 0:
                    trainingLogsBadActor.append(trainingLogs[i])
            trainingLogs = trainingLogsBadActor

        if filesToLoad == "training" or filesToLoad == "all":
            trainingLabelsBadActor = [] # Remove safe logs from training labels
            for i in range(len(trainingLabels)):
                if trainingLabels[i] != 0:
                    trainingLabelsBadActor.append(trainingLabels[i])
            trainingLabels = trainingLabelsBadActor

        if filesToLoad == "testing" or filesToLoad == "testingLogs" or filesToLoad == "all":
            testingLogsBadActor = [] # Remove safe logs from testing logs
            for i in range(len(testingLogs)):
                if testingLabels[i] != 0:
                    testingLogsBadActor.append(testingLogs[i])
            testingLogs = testingLogsBadActor

        if filesToLoad == "testing" or filesToLoad == "all":
            testingLabelsBadActor = [] # Remove safe logs from testing labels
            for i in range(len(testingLabels)):
                if testingLabels[i] != 0:
                    testingLabelsBadActor.append(testingLabels[i])
            testingLabels = testingLabelsBadActor

    if filesToLoad == "testing" or filesToLoad == "testingLogs" or filesToLoad == "all":
        testingSet = open(fileNames[2], 'r', encoding="utf8") # save the original logs unprocessed so they can be output later (to logs)
        logs = []
        for message in testingSet:
            logs.append(message)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def BinomialModelTrain(): # Train and save a support vector machines (SVM) algorithm
    ProcessLogs("all","binomial",True, True)
    clf = SVC(kernel='rbf') # Uses a radial basis function kernel
    clf.fit(trainingLogs, trainingLabels)
    predictions = clf.predict(testingLogs)
    print("Accuracy score of trained binomial model: " + str(accuracy_score(testingLabels,predictions)))
    print("f1 score of trained binomial model: " + str(f1_score(testingLabels,predictions,pos_label='safe')))
    print("precision score of trained binomial model: " + str(precision_score(testingLabels,predictions,pos_label='safe')))
    print("recall score of trained binomial model: " + str(recall_score(testingLabels,predictions,pos_label='safe')))
    pickle.dump(clf, open('SVMBinomial.sav', 'wb')) # Save the model

def MultinomialModelTrain():
    ProcessLogs("all","multinomial",False, True)
    clf = SVC(kernel='rbf') # Uses a radial basis function kernel
    clf.fit(trainingLogs, trainingLabels)
    predictions = clf.predict(testingLogs)
    print("Accuracy score of trained multinomial model: " + str(accuracy_score(testingLabels,predictions)))
    print("f1 score of trained multinomial model: " + str(f1_score(testingLabels,predictions,average='micro')))
    print("precision score of trained multinomial model: " + str(precision_score(testingLabels,predictions,average='micro')))
    print("recall score of trained multinomial model: " + str(recall_score(testingLabels,predictions,average='micro')))
    pickle.dump(clf, open('SVMMultinomial.sav', 'wb')) # Save the model

def MakePredictions(compareToLabels = False): # Writes each log and its corresponding prediction to a file called "AlgorithmOutput.txt". If compareToLabels == True, then output accuracy.
    # Loads model, and makes prediction
    global testingLabels
    # Use binomial model to predict if "safe" or "unsafe"
    ProcessLogs("testing","binomial",True,True) # Load testing logs in binomial format. Include safe logs, and load a vectorizer
    loadedModel = pickle.load(open('SVMBinomial.sav', 'rb')) 
    biPredictions = loadedModel.predict(testingLogs)
    """for i in range(len(testingLabels)):
        if testingLabels[i] == 0:
            testingLabels[i] = 'safe'
        else:
            testingLabels[i] = 'unsafe'"""
    if compareToLabels == True:
        print("Accuracy of binomial predictions: " + str(accuracy_score(testingLabels,biPredictions)))
        print("f1 of binomial predictions: " + str(f1_score(testingLabels,biPredictions,average='micro')))
        print("precision of binomial predictions: " + str(precision_score(testingLabels,biPredictions,average='micro')))
        print("recall of binomial predictions: " + str(recall_score(testingLabels,biPredictions,average='micro')))

    # Removes logs categorized as safe from testing set.
    ProcessLogs("testing","multinomial",True,True) # Load testing logs in multinomial format. Include safe logs, and load a vectorizer
    badActorLogs = []
    badActorLabels = []

    for i in range(len(biPredictions)): # Create set of logs and labels with only the logs labeled as unsafe
        if biPredictions[i] == "unsafe":
            badActorLogs.append(testingLogs[i])
            badActorLabels.append(testingLabels[i])


    # Use multinomial model to predict which bad actors unsafe logs represent
    loadedModel = pickle.load(open('SVMMultinomial.sav', 'rb'))
    multiPredictions = loadedModel.predict(badActorLogs)

    # Put multinomial predictions back into testingSet
    j = 0
    for i in range(len(biPredictions)): # Iterate through all of binomial predictions and replace unsafe predictions with their categorical prediction
        if biPredictions[i] == "unsafe":
            biPredictions[i] = multiPredictions[j]
            j += 1
    finishedPrediction = biPredictions
    if compareToLabels == True:
        print("Accuracy of binomial predictions: " + str(accuracy_score(testingLabels,finishedPrediction)))
        print("f1 of binomial predictions: " + str(f1_score(testingLabels,finishedPrediction,average='micro')))
        print("precision of binomial predictions: " + str(precision_score(testingLabels,finishedPrediction,average='micro')))
        print("recall of binomial predictions: " + str(recall_score(testingLabels,finishedPrediction,average='micro')))

    #write predictions to output file
    f = open("LogPredictions.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "AlgorithmOutput.txt" if one doesn't already exist
    frequencies = {'safe': 0, 'ws': 0, 'sql': 0, 'ddos': 0, 'ps': 0, 'ssh': 0, 'su': 0, 'unsafe':0}
    #TODO: change to output predictions in new format.
    j=0 # keeps track of the multinomial predictions position.
    for i in range(len(biPredictions)):
        if biPredictions[i] == "safe": # If log is safe, add 1 to the safe frequency.
            frequencies[biPredictions[i]] = (frequencies.get(biPredictions[i]) + 1)
        else: # If log is unsafe, add 1 to the unsafe frequency, and add 1 to the specific bad actor frequency.
            frequencies[biPredictions[i]] = (frequencies.get(biPredictions[i]) + 1)
            frequencies[multiPredictions[j]] = (frequencies.get(multiPredictions[j]) + 1)
            j += 1 # Moves to the next multinomial prediction
    for badActor in frequencies:
        frequencies[badActor] = (frequencies.get(badActor)/len(biPredictions))*100
   
    j=0 # keeps track of the multinomial predictions position.
    for i in range(len(biPredictions)): # Write every log and its corresponding prediction to file
        f.write("Prediction: ")
        if biPredictions[i] == "safe": # If log is safe, then write that to the file
            f.write(biPredictions[i]) # Write prediction to file
            f.write(" Frequency: ")
            f.write(f'{frequencies.get(biPredictions[i]):.4g}') #TODO: LEFT OFF HERE@@
        else: # If log is unsafe, then write its multinomial classification to file
            f.write(multiPredictions[j])
            j += 1
            f.write(" Frequency: ")
            f.write(f'{frequencies.get(multiPredictions[j-1]):.4g}') #TODO: LEFT OFF HERE@@
        f.write("%   log: ")
        f.write(logs[i]) # Write original log to file (not inluding '\n')
        f.write(featureExtract(logs[i]))
        f.write("\tLine Number: " + str(i+1) + "\n\n")
    f.close()

def featureExtract(line):
    features = ""
    if re.search("^\[",line): #log starts with a square bracket
       # print("This is a error.log log file")
        line = re.split("\[", line, 1)[1]
        date = re.split("(?<=.{10})\s", line, 1)[0] #gets first part of date. Will be concatenated with year later
        line = re.split("(?<=.{10})\s", line, 1)[1]
        time = re.split("\s", line, 1)[0]
        line = re.split("\s", line, 1)[1]
        date = date + " " + re.split("]", line, 1)[0] #concatenates year with the rest of the date
        line = re.split("]", line, 1)[1]
        line = re.split("\[", line, 1) [1]
        logType = re.split("\]", line, 1) [0]
        line = re.split("\[", line, 1) [1]
        IDs = re.split("\]", line, 1) [0]
        PID = re.split("\s", IDs, 1)[1]
        PID = re.split("\:", PID, 1)[0]
        TID = re.split("\s", IDs, 2)[2]
        client = ""
        if re.search("\[(?=client)", line): #ignore if brackets arent for client
            line = re.split("\[", line, 1)[1]
            line = re.split("\s", line, 1)[1]
            client = re.split("\]", line, 1)[0]
        line = re.split("\s(?=A)", line, 1)[1] #find space followed by "A" 
        errorCode = re.split("\:", line, 1)[0]
        msg = re.split("\:", line, 1)[1]
        features = "\ttime = " + time + "\n\t" + "date = " + date + "\n\t" + "log type = " + logType + "\n\t" + "PID = " + PID + "\n\t" + "TID = " + TID + "\n\t" + "client = " + client + "\n\t" + "error code = " + errorCode + "\n\t" + "message =" + msg
        
    elif re.search("^\d",line): #log starts with a digit
        #print("This is a access.log log file") #used for malicious web server access bad actor
        #log format:
        #ip - user (- if not relevant) [date & time] "GET \X"-"(if not needed)
            #else "GET \X" \d \d "http://ip/" "Browser information"
        ip = re.split("\s", line, 1)[0]
        line = re.split("\s", line, 1)[1]
        line = re.split("(?<=-)\s", line, 1)[1]
        user = re.split("\s", line, 1)[0]
        line = re.split("\s\[", line, 1)[1]
        date = re.split(":", line, 1)[0]
        line = re.split(":", line, 1)[1]
        time = re.split("]", line, 1)[0]
        msg = re.split("] ", line, 1)[1]
        features = "\tip = " + ip + "\n\t" + "user = " + user + "\n\t" + "date = " + date + "\n\t" + "time = " + time + "\n\t" + "message = " + msg

    elif re.search("^[A-Za-z]",line): #log starts with a alphabetical character
        #print("This is a auth.log log file")
        #log format:
        #month  day hour:minute:second user useraccount logtype: main information
        date = re.split("(?<=.{6})\s", line, 1)[0]
        line = re.split("(?<=.{6})\s", line, 1)[1]
        time = re.split("(?<=.{8})\s", line, 1)[0]
        line = re.split("(?<=.{8})\s", line, 1)[1]
        user = re.split("\s", line)[0]
        msg = re.split("\s", line, 1)[1]
        features = "\tdate = " + date + "\n\t" + "time = " + time + "\n\t" + "user = " + user + "\n\t" + "message = " + msg
    return features

def createBadActorSet(predictions): # Creates a new set with only unsafe logs
    #write predictions to output file
    f = open("BadActors.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "BadActors.txt" if one doesn't already exist

    frequencies = {'safe': 0, 'ws': 0, 'sql': 0, 'ddos': 0, 'ps': 0, 'ssh': 0, 'su': 0}
    
    for i in range(len(predictions)): # Write every log and its corresponding prediction to file
        if(predictions[i] == "unsafe"): # Write log to file
            f.write(logs[i]) # Write original log to file (not inluding '\n')
'''
def makeMultinomialPredictions(): # TODO: modify to fit specifically Multinomial predictions
    global trainingLogs, testingLogs, trainingLabels, testingLabels
    Tk().withdraw() # We don't want a full GUI, so keep the root window from appearing
    fileNames = []
    fileNames.append(askopenfilename(title='Multinomial Training Labels'))
    fileNames.append(askopenfilename(title='Multinomial Testing Labels'))
    with open(fileNames[0], 'r', encoding='utf8') as f:
        trainingLabels = json.load(f)
    with open(fileNames[1], 'r', encoding='utf8') as f:
        testingLabels = json.load(f)
    labelTranslation = ["safe", "ssh", "ws", "sql", "ddos", "ps", "su","unsafe"]
    for i in range(len(trainingLabels)):
        label = trainingLabels[i]
        trainingLabels[i] = labelTranslation[label]
    for i in range(len(testingLabels)):
        label = testingLabels[i]
        testingLabels[i] = labelTranslation[label]
    # Loads model, and makes prediction 
    loadedModel = pickle.load(open('SVMMultinomialModel.sav', 'rb'))
    predictions = loadedModel.predict(testingLogs)
    createBadActorSet(predictions)
    result = loadedModel.score(testingLogs, testingLabels)
    print("Loaded model accuracy: " + str(result))
    #write predictions to output file
    f = open("AlgorithmMultinomialOutput.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "AlgorithmOutput.txt" if one doesn't already exist
    frequencies = {'safe': 0, 'ws': 0, 'sql': 0, 'ddos': 0, 'ps': 0, 'ssh': 0, 'su': 0, 'unsafe':0}
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
        f.write(featureExtract(logs[i]))
        f.write("\tLine Number: " + str(i+1) + "\n\n")
    f.close()
'''

def makeBadActorSet(): # Makes a text file that contains bad actor training and testing logs and labels.
    Tk().withdraw() # We don't want a full GUI, so keep the root window from appearing
    fileNames = []
    fileNames.append(askopenfilename(title='Training Set'))
    fileNames.append(askopenfilename(title='Multinomial Training Labels'))
    fileNames.append(askopenfilename(title='Testing Set'))
    fileNames.append(askopenfilename(title='Multinomial Testing Labels'))
    fileNames.append(askopenfilename(title='vectorizerModel'))

    global trainingLogs, testingLogs, trainingLabels, testingLabels
    
    with open(fileNames[4], 'rb') as f: # Load vectorizer
        vectorizer = pickle.load(f)

    with open(fileNames[0], 'r', encoding='utf8') as f: #
        vectors = vectorizer.transform(f)
        dense = vectors.todense()
        trainingLogs = np.asarray(dense)

    with open(fileNames[1], 'r', encoding='utf8') as f:
        trainingLabels = json.load(f)

    with open(fileNames[2], 'r', encoding='utf8') as f:
        vectors = vectorizer.transform(f)
        dense = vectors.todense()
        testingLogs = np.asarray(dense)

    with open(fileNames[3], 'r', encoding='utf8') as f:
        testingLabels = json.load(f)
    
    # Returns position in file to original file
    testingSet = open(fileNames[2], 'r', encoding="utf8")
    testingLogs = []
    for message in testingSet: # Create a list of labels for the testing set.
        testingLogs.append(message)
    trainingSet = open(fileNames[2], 'r', encoding="utf8")
    trainingLogs = []
    for message in trainingSet: # Create a list of labels for the testing set.
        trainingLogs.append(message)

    # Write exclusively bad actor training labels to their own text file
    f = open("TrainingLabelsBadActor.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "BadActorLogs.txt" if one doesn't already exist
    firstWrite = True # used to determine if this is the first label that has been written. If it is, then don't put a comma before it to format list properly.
    f.write("[")
    for i in range(len(trainingLogs)):
        if trainingLabels[i] != 0:
            if firstWrite == False:
                f.write("," + str(trainingLabels[i]))
            else: # firstWrite == True, don't write comma before
                f.write(str(trainingLabels[i]))   
                firstWrite = False     
    f.write("]")
    f.close()

    # Write exclusively bad actor training logs to their own text file
    f = open("TrainingLogsBadActor.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "BadActorLabels.txt" if one doesn't already exist
    for i in range(len(trainingLogs)):
        if trainingLabels[i] != 0:
            f.write(trainingLogs[i])
    f.close()

    # Write exclusively bad actor testing labels to their own text file
    f = open("TestingLabelsBadActor.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "BadActorLogs.txt" if one doesn't already exist
    firstWrite = True # used to determine if this is the first label that has been written. If it is, then don't put a comma before it to format list properly.
    f.write("[")
    for i in range(len(trainingLogs)):
        if trainingLabels[i] != 0:
            if firstWrite == False:
                f.write("," + str(trainingLabels[i]))
            else: # firstWrite == True, don't write comma before
                f.write(str(testingLabels[i]))      
                firstWrite = False  
    f.write("]")
    f.close()

    # Write exclusively bad actor testing logs to their own text file
    f = open("TestingLogsBadActor.txt", "w") #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "BadActorLabels.txt" if one doesn't already exist
    for i in range(len(trainingLogs)):
        if trainingLabels[i] != 0:
            f.write(testingLogs[i])
    f.close()

#makeBadActorSet() # Makes a text file that contains bad actor training and testing logs and labels.

#ProcessLogs()


#BinomialModelTrain()
#MultinomialModelTrain()
MakePredictions(True)


#makeMultinomialPredictions()
