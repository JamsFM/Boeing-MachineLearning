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
    #filesToLoad options: "all", "training", "testing" (logs and labels), "testingLogs". Loads file names
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
    else: # Load vectorizer
        with open(fileNames[4], 'rb') as f: 
            vectorizer = pickle.load(f)
    if filesToLoad == "training" or filesToLoad == "all": # Load trainingLogs
        with open(fileNames[0], 'r', encoding='utf8') as f: #
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            trainingLogs = np.asarray(dense)
    if filesToLoad == "training" or filesToLoad == "all": # Load trainingLabels
        with open(fileNames[1], 'r', encoding='utf8') as f:
            trainingLabels = json.load(f)
    if filesToLoad == "testing" or filesToLoad == "testingLogs" or filesToLoad == "all": # Load testingLogs
        with open(fileNames[2], 'r', encoding='utf8') as f:
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            testingLogs = np.asarray(dense)
    if filesToLoad == "testing" or filesToLoad == "all": # Load testingLabels
        with open(fileNames[3], 'r', encoding='utf8') as f:
            testingLabels = json.load(f)

    if cardinality == "multinomial": # Translate labels to each bad actor label
        labelTranslation = ["safe", "ssh", "ws", "sql", "ddos", "ps", "su","unsafe"]
    elif cardinality == "binomial": # Translate labels to binary classification
        labelTranslation = ["safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe"]
    else:
        print("Entered invalid cardinality to process logs function. Enter \"binomial\" or \"multinomial\".")
        return

    # Change the numeric representation of the bad actors to the string representation.
    if filesToLoad == "training" or filesToLoad == "all":
        for i in range(len(trainingLabels)): 
            trainingLabels[i] = labelTranslation[trainingLabels[i]]
    if filesToLoad == "testing" or filesToLoad == "all":
        for i in range(len(testingLabels)):
            testingLabels[i] = labelTranslation[testingLabels[i]]
    
    if not includeSafeLogs: 
        # Remove all safe logs from training and testing sets
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

    # Save the original testing logs unprocessed so they can be output later (to logs)
    if filesToLoad == "testing" or filesToLoad == "testingLogs" or filesToLoad == "all":
        testingSet = open(fileNames[2], 'r', encoding="utf8") 
        logs = []
        for message in testingSet:
            logs.append(message)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def BinomialModelTrain(): # Train and save a binomial support vector machines (SVM) algorithm
    ProcessLogs("all","binomial",True, True)
    clf = LogisticRegression(solver = 'liblinear') #'lbfgs' 'liblinear', might be more accurate just slower --------------------
    clf.fit(trainingLogs, trainingLabels) # Train SVM algorithm using trainingLogs and trainingLabels
    predictions = clf.predict(testingLogs) # Predict testingLogs with newly trained model
    # Output the accuracy metrics of the model
    print("Accuracy score of trained binomial model: " + str(accuracy_score(testingLabels,predictions)))
    print("f1 score of trained binomial model: " + str(f1_score(testingLabels,predictions,pos_label='safe')))
    print("precision score of trained binomial model: " + str(precision_score(testingLabels,predictions,pos_label='safe')))
    print("recall score of trained binomial model: " + str(recall_score(testingLabels,predictions,pos_label='safe')))
    pickle.dump(clf, open('LRBinomial.sav', 'wb')) # Save the model as 'LRBinomial.sav'

def MultinomialModelTrain(): # Train and save a multinomial support vector machines (SVM) algorithm
    ProcessLogs("all","multinomial",False, True)
    clf = LogisticRegression(multi_class='multinomial', solver = 'lbfgs') #'lbfgs' 'liblinear', might be more accurate just slower --------------------
    clf.fit(trainingLogs, trainingLabels) # Train SVM algorithm using trainingLogs and trainingLabels
    predictions = clf.predict(testingLogs) # Predict testingLogs with newly trained model
    # Output the accuracy metrics of the model
    print("Accuracy score of trained multinomial model: " + str(accuracy_score(testingLabels,predictions)))
    print("f1 score of trained multinomial model: " + str(f1_score(testingLabels,predictions,average='micro')))
    print("precision score of trained multinomial model: " + str(precision_score(testingLabels,predictions,average='micro')))
    print("recall score of trained multinomial model: " + str(recall_score(testingLabels,predictions,average='micro')))
    pickle.dump(clf, open('LRMultinomial.sav', 'wb')) # Save the model as 'LRMultinomial.sav'

def MakePredictions(compareToLabels = False): # Writes each log and its corresponding prediction to a file called "AlgorithmOutput.txt". If compareToLabels == True, then output accuracy
    # Loads model, and makes prediction
    
    global testingLabels
    # Use binomial model to predict if "safe" or "unsafe"
    ProcessLogs("testing","binomial",True,True) # Load testing logs in binomial format. Include safe logs, and load a vectorizer
    loadedModel = pickle.load(open('LRBinomial.sav', 'rb')) # Load binomial model
    biPredictions = loadedModel.predict(testingLogs) # Make predictions on testingLogs (binomially)
    
    if compareToLabels == True: # Output the accuracy metrics of the model
        print("Accuracy of binomial predictions: " + str(accuracy_score(testingLabels,biPredictions)))
        print("f1 of binomial predictions: " + str(f1_score(testingLabels,biPredictions,average='micro')))
        print("precision of binomial predictions: " + str(precision_score(testingLabels,biPredictions,average='micro')))
        print("recall of binomial predictions: " + str(recall_score(testingLabels,biPredictions,average='micro')))

    # Load testing logs, then remove logs categorized as safe from testing set. Predict which bad actors remaining bad actor logs represent using multinomial model
    ProcessLogs("testing","multinomial",True,True) # Load testing logs in multinomial format. Include safe logs, and load a vectorizer
    badActorLogs = []
    badActorLabels = []
    for i in range(len(biPredictions)): # Create set of logs and labels with only the logs labeled as unsafe
        if biPredictions[i] == "unsafe":
            badActorLogs.append(testingLogs[i])
            badActorLabels.append(testingLabels[i])
    # Use multinomial model to predict which bad actors unsafe logs represent
    loadedModel = pickle.load(open('LRMultinomial.sav', 'rb')) # Load multinomial model
    multiPredictions = loadedModel.predict(badActorLogs)

    # Combine multinomial predictions and binomial predictions into one set
    j = 0
    for i in range(len(biPredictions)): # Iterate through all of binomial predictions and replace unsafe predictions with their categorical prediction
        if biPredictions[i] == "unsafe":
            biPredictions[i] = multiPredictions[j]
            j += 1
    finishedPrediction = biPredictions # finishedPrediction is the final set of predictions
    if compareToLabels == True: # Output the accuracy metrics of the model
        print("Accuracy of overall predictions: " + str(accuracy_score(testingLabels,finishedPrediction)))
        print("f1 of overall predictions: " + str(f1_score(testingLabels,finishedPrediction,average='micro')))
        print("precision of overall predictions: " + str(precision_score(testingLabels,finishedPrediction,average='micro')))
        print("recall of overall predictions: " + str(recall_score(testingLabels,finishedPrediction,average='micro')))

    # Write predictions to output file
    f = open("LogPredictions.txt", "w") # "w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "AlgorithmOutput.txt" if one doesn't already exist
    
    # Find frequency of each prediction in testing set
    frequencies = {'safe': 0, 'ws': 0, 'sql': 0, 'ddos': 0, 'ps': 0, 'ssh': 0, 'su': 0, 'unsafe':0}
    j=0 # Keeps track of the multinomial predictions position.
    for i in range(len(biPredictions)):
        if biPredictions[i] == "safe": # If log is safe, add 1 to the safe frequency.
            frequencies[biPredictions[i]] = (frequencies.get(biPredictions[i]) + 1)
        else: # If log is unsafe, add 1 to the unsafe frequency, and add 1 to the specific bad actor frequency.
            # Keeps track of frequency of unsafe as a whole, and frequency of specific bad actors.
            frequencies[biPredictions[i]] = (frequencies.get(biPredictions[i]) + 1)
            frequencies[multiPredictions[j]] = (frequencies.get(multiPredictions[j]) + 1)
            j += 1 # Moves to the next multinomial prediction
    for badActor in frequencies:
        frequencies[badActor] = (frequencies.get(badActor)/len(biPredictions))*100
   
    j=0 # Keeps track of the multinomial predictions position.
    for i in range(len(biPredictions)): # Write every log and its corresponding prediction to file
        f.write("Prediction: ")
        if biPredictions[i] == "safe": # If log is safe, then write that to the file
            f.write(biPredictions[i]) # Write prediction to file
            f.write(" Frequency: ")
            f.write(f'{frequencies.get(biPredictions[i]):.4g}')
        else: # If log is unsafe, then write its multinomial classification to file
            f.write(multiPredictions[j])
            j += 1
            f.write(" Frequency: ")
            f.write(f'{frequencies.get(multiPredictions[j-1]):.4g}') # Only output the specific bad actor's frequency, not overall bad actor frequency
        f.write("%   log: ")
        f.write(logs[i]) # Write original log to file
        f.write(featureExtract(logs[i])) # Uses regex to output the components of the log
        f.write("\tLine Number: " + str(i+1) + "\n\n") # Outputs which log is being output
    f.close()

def featureExtract(line): # Uses regex to output the components of the log
    features = ""
    if re.search("^\[",line): # Log starts with a square bracket
        # This is a error.log log file. Following regex is specific to this log file in Linux
        line = re.split("\[", line, 1)[1]
        date = re.split("(?<=.{10})\s", line, 1)[0] # Gets first part of date. Will be concatenated with year later
        line = re.split("(?<=.{10})\s", line, 1)[1]
        time = re.split("\s", line, 1)[0]
        line = re.split("\s", line, 1)[1]
        date = date + " " + re.split("]", line, 1)[0] # Concatenates year with the rest of the date
        line = re.split("]", line, 1)[1]
        line = re.split("\[", line, 1) [1]
        logType = re.split("\]", line, 1) [0]
        line = re.split("\[", line, 1) [1]
        IDs = re.split("\]", line, 1) [0]
        PID = re.split("\s", IDs, 1)[1]
        PID = re.split("\:", PID, 1)[0]
        TID = re.split("\s", IDs, 2)[2]
        client = ""
        if re.search("\[(?=client)", line): # Ignore if brackets aren't for client
            line = re.split("\[", line, 1)[1]
            line = re.split("\s", line, 1)[1]
            client = re.split("\]", line, 1)[0]
        line = re.split("\s(?=A)", line, 1)[1] # Find space followed by "A" 
        errorCode = re.split("\:", line, 1)[0]
        msg = re.split("\:", line, 1)[1]
        features = "\ttime = " + time + "\n\t" + "date = " + date + "\n\t" + "log type = " + logType + "\n\t" + "PID = " + PID + "\n\t" + "TID = " + TID + "\n\t" + "client = " + client + "\n\t" + "error code = " + errorCode + "\n\t" + "message =" + msg
        
    elif re.search("^\d",line): # Log starts with a digit
        # This is a access.log log file, used for malicious web server access bad actor
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

    elif re.search("^[A-Za-z]",line): # Log starts with a alphabetical character
        # This is a auth.log log file
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
    f = open("BadActors.txt", "w") # "w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "BadActors.txt" if one doesn't already exist

    frequencies = {'safe': 0, 'ws': 0, 'sql': 0, 'ddos': 0, 'ps': 0, 'ssh': 0, 'su': 0}
    
    for i in range(len(predictions)): # Write every log and its corresponding prediction to file
        if(predictions[i] == "unsafe"): # Write log to file
            f.write(logs[i]) # Write original log to file (not inluding '\n')


BinomialModelTrain()
MultinomialModelTrain()
MakePredictions(True)
