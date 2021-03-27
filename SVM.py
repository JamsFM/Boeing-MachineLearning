import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import random

logs = open(os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))), 'BadActorLogs.txt'), 'r', encoding="utf8") #Opens logs.txt (has to be in same directory as python script)
#training labels for combined logs

#trainingLabels = ["safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","safe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","unsafe","safe","safe","safe","safe","unsafe","safe","safe","unsafe","unsafe","unsafe","safe","safe","safe","unsafe","unsafe","unsafe","safe","unsafe","unsafe","unsafe"]

#training labels for bad actor logs
trainingLabels = ["Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","Unauthorized Web Server Logins","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","malicious webserver access","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","ddos","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","port scan","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized login","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","unauthorized superuser privileges","failed login","failed login","failed login","unauthorized login","unauthorized login","unauthorized login"]

# Vectorizes log file into numerical array
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(logs)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
for i in range(len(denselist)):
        denselist[i].insert(0,trainingLabels[i])

random.shuffle(denselist)

df = pd.DataFrame(denselist)
#print(feature_names)
#print(df)

#import data
balance_data = df
# Printing the dataset shape 
#print ("Dataset Length: ", len(balance_data)) 
#print ("Dataset Shape: ", balance_data.shape)
# Printing the dataset obseravtions 
#print ("Dataset: ",balance_data.head())


from sklearn.model_selection import train_test_split 
# Function to split the dataset 
def splitdataset(balance_data): 
  
    # Separating the target variable 
    X = balance_data.values[:, 1:] 
    Y = balance_data.values[:, 0] 
  
    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2) 
      
    return X, Y, X_train, X_test, y_train, y_test 

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def modeltrain():
    X, Y, x_train, x_test, y_train, y_test = splitdataset(df)

    clf = SVC(kernel='rbf', degree=8) #degree is only used for poly kernels
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print("Accuracy score of sklearn model: " + str(accuracy_score(y_test,y_pred)))
    #Save the model
    pickle.dump(clf, open('SVMModel.sav', 'wb'))

#modeltrain()

def loadmodel():
    loadedmodel = pickle.load(open('SVMModel.sav', 'rb'))
    X, Y, x_train, x_test, y_train, y_test = splitdataset(df)
    result = loadedmodel.score(x_test, y_test)
    print("loaded model accuracy: " + str(result))

loadmodel()
