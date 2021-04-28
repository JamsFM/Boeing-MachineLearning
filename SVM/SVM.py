import os

import numpy as np #TODO: Check if these libraries are needed
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import random

#for labels (not being used atm, used with file that has an array)
import json

def ProcessLogs(): # Makes each log into an array with the first element being the logs label, and the remaining elements being the tfidf representation of the log.
    """ #TODO: This is the old method to import logs and labels. May remove if unneeded, but I'm leaving this for now incase it is helpful later.
    # Vectorizes log file into numerical array
    logs = open(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), 'universalSet.txt'), 'r', encoding="utf8") #Opens logs.txt (has to be in same directory as python script)

    # Training labels for multinomial universal set
    trainingLabels = [0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 4, 4, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 6, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 6, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 6, 1, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 6, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 6, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 6, 0, 3, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 0, 5, 6, 6, 3, 0, 0, 4, 0, 0, 6, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 6, 0, 0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 5, 0, 6, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 2, 0, 0, 6, 0, 0, 0, 0, 6, 5, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 5, 0, 0, 5, 0, 3, 4, 0, 5, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 5, 2, 1, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0, 3, 0, 0, 4, 0, 6, 0, 0, 0, 0, 5, 3, 0, 4, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 2, 4, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 2, 3, 0, 0, 1, 1, 6, 0, 2, 0, 0, 6, 0, 0, 6, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 5, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 6, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 4, 0, 0, 0, 0, 0, 2, 0, 6, 0, 2, 0, 3, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 0, 0, 0, 5, 6, 0, 1, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 2, 6, 0, 0, 2, 0, 4, 0, 0, 0, 5, 0, 0, 0, 1, 0, 1, 0, 3, 0, 0, 0, 1, 3, 5, 0, 6, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 0, 0, 2, 0, 0, 0, 0, 2, 0, 5, 0, 6, 4, 0, 0, 0, 0, 5, 0, 0, 5, 0, 1, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 4, 4, 1, 0, 0, 0, 0, 2, 5, 0, 0, 4, 2, 0, 0, 0, 1, 0, 0, 5, 0, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3, 0, 1, 0, 6, 5, 6, 0, 6, 0, 0, 5, 0, 0, 3, 0, 4, 0, 3, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0]

    # Training labels for binomial universal set
    #trainingLabels = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    # Training labels for bad actor logs@@@@@@
    #trainingLabels = ["Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","failed login","failed login","failed login","unauthorized login","unauthorized login","unauthorized login"]

    #for i in trainingLabels
    #    if trainingLabels[i]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(logs)
    featureNames = vectorizer.get_feature_names()
    dense = vectors.todense()
    denseList = dense.tolist()
    for i in range(len(denseList)):
        denseList[i].insert(0,trainingLabels[i])
    #random.shuffle(denselist) # TODO: resolve issue with printing output being messed up by randomizing order
    processedLogs = pd.DataFrame(denseList)
    """

    from tkinter import Tk     # from tkinter import Tk for Python 3.x
    from tkinter.filedialog import askopenfilename


    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    logFilename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

    global logSet
    logSet = open(logFilename, 'r', encoding='utf8') #Opens logs.txt (has to be in same directory as python script)

    # Fits TfidfVectorizer with logs
    vectorizer = TfidfVectorizer()
    vectorizer.fit(logSet)
    print("Vectorizer Trained.")

    # Returns position in file to original file
    logSet.seek(0)
    global logs 
    logs = []
    for message in logSet:
        logs.append(message)
    logSet.seek(0)

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    labelsFilename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

    with open(labelsFilename, 'r', encoding='utf8') as file:
        logLabels = np.array(json.load(file))

    logLabelStrings = []
    labelTranslation = ["safe", "ssh", "ws", "sql", "ddos", "ps", "su"]
    for i in range(len(logLabels)):
        logLabelStrings.append(labelTranslation[logLabels[i]])
    # Transforms logs into tf-idf matrix
    # and outputs as array
    vectors = vectorizer.transform(logSet)
    dense = vectors.todense()
    logSet = np.asarray(dense)
    
    # Printing the dataset shape 
    #print ("Dataset Length: ", len(processedLogs)) 
    #print ("Dataset Shape: ", processedLogs.shape)
    # Printing the dataset obseravtions 
    #print ("Dataset: ",processedLogs.head())
    #print(feature_names)
    #print(df)
    return logSet, logLabelStrings

from sklearn.model_selection import train_test_split 
# Function to split the dataset 
def SplitDataset():
    processedLogs = ProcessLogs()
    # Separating the target variable 
    tfidfLogs = processedLogs[0]
    Labels = processedLogs[1]

    # Splitting the dataset into train and test 
    xTrain, xTest, yTrain, yTest = train_test_split(tfidfLogs, Labels, test_size = 0.2) 
    
    return tfidfLogs, Labels, xTrain, xTest, yTrain, yTest 

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def ModelTrain(): # Train and save a support vector machines (SVM) algorithm
    tfidfLogs, Labels, xTrain, xTest, yTrain, yTest = SplitDataset()

    clf = SVC(kernel='rbf') # Uses a radial basis function kernel
    clf.fit(xTrain,yTrain)
    yPred = clf.predict(xTest)
    print("Accuracy score of trained sklearn model: " + str(accuracy_score(yTest,yPred)))
    pickle.dump(clf, open('SVMModel.sav', 'wb')) # Save the model

def MakePredictions(): # Writes each log and its corresponding prediction to a file called "AlgorithmOutput.txt"
    # Loads model, and makes prediction
    loadedModel = pickle.load(open('SVMModel.sav', 'rb'))
    tfidfLogs, Labels, xTrain, xTest, yTrain, yTest = SplitDataset()
    predictions = loadedModel.predict(tfidfLogs)
    result = loadedModel.score(tfidfLogs, Labels)
    print("Loaded model accuracy: " + str(result))

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

ModelTrain()
MakePredictions()
