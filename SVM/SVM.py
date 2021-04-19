import os

import numpy as np #TODO: Check if these libraries are needed
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import random

def ProcessLogs(): # Makes each log into an array with the first element being the logs label, and the remaining elements being the tfidf representation of the log.
    # Vectorizes log file into numerical array
    logs = open(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), 'BadActorLogs.txt'), 'r', encoding="utf8") #Opens logs.txt (has to be in same directory as python script)

    # Training labels for combined logs
    #trainingLabels = ["safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","unsafe","safe","safe","unsafe","unsafe","unsafe","safe","safe","safe","unsafe","unsafe","unsafe","safe","unsafe","unsafe","unsafe"]

    # Training labels for bad actor logs
    trainingLabels = ["Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","failed login","failed login","failed login","unauthorized login","unauthorized login","unauthorized login"]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(logs)
    featureNames = vectorizer.get_feature_names()
    dense = vectors.todense()
    denseList = dense.tolist()
    for i in range(len(denseList)):
        denseList[i].insert(0,trainingLabels[i])
    #random.shuffle(denselist) # TODO: resolve issue with printing output being messed up by randomizing order
    processedLogs = pd.DataFrame(denseList)

    # Printing the dataset shape 
    #print ("Dataset Length: ", len(processedLogs)) 
    #print ("Dataset Shape: ", processedLogs.shape)
    # Printing the dataset obseravtions 
    #print ("Dataset: ",processedLogs.head())
    #print(feature_names)
    #print(df)
    return processedLogs

from sklearn.model_selection import train_test_split 
# Function to split the dataset 
def SplitDataset():
    processedLogs = ProcessLogs()
    # Separating the target variable 
    tfidfLogs = processedLogs.values[:, 1:]
    Labels = processedLogs.values[:, 0] 

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
    logs = open(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), 'unOrgLogs.txt'), 'r', encoding="utf8") # Opens logs.txt (has to be in same directory as python script)
    frequencies = {'safe': 0, 'Unauthorized Web Server Logins': 0, 'malicious webserver access': 0, 'ddos': 0, 'port scan': 0, 'ssh': 0, 'unauthorized superuser privileges': 0}
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
        f.write(logs.readline().rstrip('\n')) # Write original log to file (not inluding '\n')
        f.write("\n")
    f.close()

#ModelTrain()
MakePredictions()
