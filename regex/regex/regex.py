import re
import os #for opening file in same directory as python script

logs = open(os.path.join(sys.path[0], "logs.txt"), 'r')

for line in logs:
    if re.search("^\[",line): #log starts with a square bracket
       # print("This is a error.log log file")
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
        print("message =" + msg )
        
        
    elif re.search("^\d",line): #log starts with a digit
        #print("This is a access.log log file") #used for malicious web server access bad actor
        #log format:
        #ip - user (- if not relevant) [date & time] "GET \X"-"(if not needed)
            #else "GET \X" \d \d "http://ip/" "Browser information"
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
        print("message = " + msg)
        

    elif re.search("^[A-Za-z]",line): #log starts with a alphabetical character
        #print("This is a auth.log log file")
        #log format:
        #month  day hour:minute:second user useraccount logtype: main information
        date = re.split("(?<=.{6})\s", line, 1)[0]
        print("date = " + date)
        line = re.split("(?<=.{6})\s", line, 1)[1]
        time = re.split("(?<=.{8})\s", line, 1)[0]
        print("time = " + time)
        line = re.split("(?<=.{8})\s", line, 1)[1]
        user = re.split("\s", line)[0]
        print("user = " + user)
        msg = re.split("\s", line, 1)[1]
        print("message = " + msg)

logs.close()

#print("End of program")

#access.log ip, date time, use quotes to determine next two sections
#error.log date time, basic log type, process & thread id (possibly client also), AH0 something, main log information
#other logs ???

#auth.log: date time, user account, log type, (separated by :) main information
#error.log starts with '[', access.log starts with a digit, auth.log starts with a letter (first letter of month)

#use \d to detect any digit
#use \D to detect any non-digit
#[A-Za-z]

#implementing term frequency-inverse document frequency
#add all features of entire log file to a dictionary (make sure there are no duplicates) Make dictionary with a trie
#Trie references: https://pygtrie.readthedocs.io/en/latest/ , https://en.wikipedia.org/wiki/Trie
#Take each feature of a log and add 1 frequency to its occurence in the dictionary


#make an 2 arrays, 1 with the content of the string at nth position, and other with number of occurences 
#at nth position
#now have term frequency

#@@@@@@@@@@@@@@Look for better data structures

#print("End of program")

#access.log ip, date time, use quotes to determine next two sections
#error.log date time, basic log type, process & thread id (possibly client also), AH0 something, main log information
#other logs ???

#auth.log: date time, user account, log type, (separated by :) main information
#error.log starts with '[', access.log starts with a digit, auth.log starts with a letter (first letter of month)

#use \d to detect any digit
#use \D to detect any non-digit
#[A-Za-z]

"""
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@End of parameterization, start of machine learning @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
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