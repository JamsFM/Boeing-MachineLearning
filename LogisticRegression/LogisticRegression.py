import os

import numpy as np
import pandas as pd

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
    processedLogs = pd.DataFrame(denseList)

    return processedLogs


#NOTHING WORKING YET, JUST THROWING CODE AT A WALL FOR SPAGHETTI CODE (NOT AL DENTE YET!)
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#use this one   https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a


from sklearn.model_selection import train_test_split 

# Function to split the dataset 
def SplitDataset():
    processedLogs = ProcessLogs()
    # Separating the target variable 
    tfidfLogs = processedLogs.values[:, 1:]
    Labels = processedLogs.values[:, 0] 

    # Splitting the dataset into train and test 
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
    
    return tfidfLogs, Labels, xTrain, xTest, yTrain, yTest 


from sklearn.linear_model import LogisticRegression
#from sklearn.feature_selection import RFE
#import seaborn as sns

def ModelTrain():
    tfidfLogs, Labels, xTrain, xTest, yTrain, yTest = SplitDataset()
    
    logRegrModel = LogisticRegression(solver = 'lbfgs') # 'liblinear' ,ight be more accurate just slower
    logRegrModel.fit(xTrain, yTrain)
    predictions = logisticRegr.predict(xTest)

    # Use score method to get accuracy of model
    score = logisticRegr.score(xTest, yTest) # correct predictions / total number of data points
    print("Score: ",score)

    #rfe = RFE(logreg, 20)
    #rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
    #print(rfe.support_)
    #print(rfe.ranking_)


if __name__=='__main__':
    ProcessLogs()
    ModelTrain()
