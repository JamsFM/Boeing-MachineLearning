import re

logs = open('C:/Users/Test/Desktop/Nicholas Falter/School (Current)/Senior Design/Code/logs.txt', 'r') #TODO: INSERT LOCATION OF LOG FILE HERE
dictionary = []
baseLog = [] #Will be added to as dictionary is populated. Has a 0 for each dictionary member.
#def inDictionary(feature):
#    for member in dictionary:
#        if member == feature:
#            return 1
#    return 0

"""
#this method counts the number of occurences of features in the entire file (because baggedLog and baseLog refer to the same thing. This is poor code)
#TODO: Remove or improve this function.
def bag(log): #implement bagofwords
    baggedLog = baseLog #start with a vector of 0's (n zeros, n dictionary members)
    for feature in log: #look at each feature in the log
        count = 0 #keep track of which member of the dictionary is being compared
        inDictionary = 0 #represents if the feature is found in the dictionary
        for member in dictionary: #search through dictionary
            if member == feature: #if feature is in dictionary
                baggedLog[count] += 1 #add 1 to the frequency of the feature
                inDictionary = 1
                break #exit search for feature in dictionary
            count += 1
        if not inDictionary: #if feature isn't in dictionary
            dictionary.append(feature) #add it to the dictionary
            baseLog.append(1) #add a zero for the new feature
    return baggedLog
"""

def bag(log): #implement bagofwords
    baggedLog = baseLog.copy() #start with a vector of 0's (n zeros, n dictionary members)
    for feature in log: #look at each feature in the log
        count = 0 #keep track of which member of the dictionary is being compared
        inDictionary = 0 #represents if the feature is found in the dictionary
        for member in dictionary: #search through dictionary
            if member == feature: #if feature is in dictionary
                baggedLog[count] += 1 #add 1 to the frequency of the feature
                inDictionary = 1
                break #exit search for feature in dictionary
            count += 1
        if not inDictionary: #if feature isn't in dictionary
            dictionary.append(feature) #add it to the dictionary
            baseLog.append(0) #add a zero for the new feature
            baggedLog.append(1) #Count the new feature
    return baggedLog

for line in logs:
    if re.search("^\[",line): #log starts with a square bracket, indicating this is an error.log log
        line = re.split("\[", line, 1)[1]
        date = re.split("(?<=.{10})\s", line, 1)[0] #gets first part of date. Will be concatenated with year later
        line = re.split("(?<=.{10})\s", line, 1)[1]
        time = re.split("\s", line, 1)[0]
        print("time = " + time)
        line = re.split("\s", line, 1)[1]
        date = date + " " + re.split("]", line, 1)[0] #concatenates year with the rest of the date
        print("date = " + date)
        line = re.split("]", line, 1)[1]
        line = re.split("\[", line, 1) [1]
        logType = re.split("\]", line, 1) [0]
        print("log type = " + logType)
        line = re.split("\[", line, 1) [1]
        IDs = re.split("\]", line, 1) [0]
        PID = re.split("\s", IDs, 1)[1]
        PID = re.split("\:", PID, 1)[0]
        print("PID = " + PID)
        TID = re.split("\s", IDs, 2)[2]
        print("TID = " + TID)
        if re.search("\[(?=client)", line): #ignore if brackets arent for client
            line = re.split("\[", line, 1)[1]
            line = re.split("\s", line, 1)[1]
            client = re.split("\]", line, 1)[0]
            print("client = " + client)
        line = re.split("\s(?=A)", line, 1)[1] #find space followed by "A" 
        errorCode = re.split("\:", line, 1)[0]
        print("error code = " + errorCode)
        msg = re.split("\:", line, 1)[1]
        print("message =" + msg, end = "" ) #TODO: remove \n from the end of msg
        log = [time, date, logType, PID, TID, errorCode, msg]
        baggedLog = bag(log)
        print("bagged log: ", end = "")
        print(baggedLog)
        print("")
        
    elif re.search("^\d",line): #log starts with a digit, indicating this is an access.log log
        #used for malicious web server access bad actor
        ip = re.split("\s", line, 1)[0]
        print("ip = " + ip)
        line = re.split("\s", line, 1)[1]
        line = re.split("(?<=-)\s", line, 1)[1]
        user = re.split("\s", line, 1)[0]
        print("user = " + user)
        line = re.split("\s\[", line, 1)[1]
        date = re.split(":", line, 1)[0]
        print("date = " + date)
        line = re.split(":", line, 1)[1]
        time = re.split("]", line, 1)[0]
        print("time = " + time)
        msg = re.split("] ", line, 1)[1]
        print("message =" + msg, end = "" ) #TODO: remove \n from the end of msg
        log = [ip, user, date, time, msg]
        baggedLog = bag(log)
        print("bagged log: ", end = "")
        print(baggedLog)
        print("")

    elif re.search("^[A-Za-z]",line): #log starts with a alphabetical character, indicating this is an auth.log log
        date = re.split("(?<=.{6})\s", line, 1)[0]
        print("date = " + date)
        line = re.split("(?<=.{6})\s", line, 1)[1]
        time = re.split("(?<=.{8})\s", line, 1)[0]
        print("time = " + time)
        line = re.split("(?<=.{8})\s", line, 1)[1]
        user = re.split("\s", line)[0]
        print("user = " + user)
        msg = re.split("\s", line, 1)[1]
        print("message =" + msg, end = "" ) #TODO: remove \n from the end of msg
        log = [date, time, user, msg]
        baggedLog = bag(log)
        print("bagged log: ", end = "")
        print(baggedLog)
        print("")

logs.close()

print("Dictionary: ")
print(dictionary)

#implementing term frequency-inverse document frequency
#add all features of entire log file to a dictionary (make sure there are no duplicates) could make dictionary with a trie
#Trie references: https://pygtrie.readthedocs.io/en/latest/ , https://en.wikipedia.org/wiki/Trie

"""
#@@@@@@@@@@@@@@End of parameterization, start of machine learning (code taken from: https://www.geeksforgeeks.org/decision-tree-implementation-python/ ) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
"""
"""
# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv( 
'https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/balance-scale/balance-scale.data', 
    sep= ',', header = None) 
      
    # Printing the dataswet shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
"""
"""
# Function to split the dataset 
def splitdataset(balance_data): 
  
    # Separating the target variable 
    X = balance_data.values[:, 1:5] 
    Y = balance_data.values[:, 0] 
  
    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 


# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
  
# Driver code 
def main(): 
      
    # Building Phase 
    data = parameterizedLogs
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
      
      
# Calling main function 
if __name__=="__main__": 
    main() 
"""