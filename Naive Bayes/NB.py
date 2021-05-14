import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from tkinter import Tk
import tkinter.filedialog as fd
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def readFiles():

    global trainingVectors, testingVectors, trainingLabels, testingLabels, logs

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

    trainingLogsFile = fd.askopenfilename(title='Select Training Logs File')
    trainingLabelsFile = fd.askopenfilename(title='Select Training Labels File')
    testingLogsFile = fd.askopenfilename(title='Select Testing Logs File')
    testingLabelsFile = fd.askopenfilename(title='Select Testing Labels File')

    vectorizerFilename = fd.askopenfilename(title='Select Vectorizer Mode File')

    with open(vectorizerFilename, 'rb') as f:
        vectorizer = pickle.load(f)

    with open(trainingLogsFile, 'r', encoding='utf8') as f:
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            trainingVectors = np.asarray(dense)

    with open(testingLogsFile, 'r', encoding='utf8') as f:
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            testingVectors = np.asarray(dense)
            f.seek(0)
            logs = []
            for message in f:
                logs.append(message)

    with open(trainingLabelsFile, 'r', encoding='utf8') as f:
            trainingLabels = np.asarray(json.load(f))

    with open(testingLabelsFile, 'r', encoding='utf8') as f:
            testingLabels = np.asarray(json.load(f))

def BiModelTrain():
    global model
    
    # Creates and fits model with training logs and labels
    model = MultinomialNB()

    model.fit(trainingVectors, trainingLabels)

    print("Model Fitted w/: {} logs.".format(len(trainingLabels)))
    
    # Saves model with pickle.dump()
    with open('muNB.pickel', 'wb') as f:
        pickle.dump(model, f)

def MuModelTrain():
    global model

    # Creates and fits model with training logs and labels
    model = GaussianNB()

    model.fit(trainingVectors, trainingLabels)

    print("Model Fitted w/: {} logs.".format(len(trainingLabels)))
    
    # Saves model with pickle.dump()
    with open('biNB.pickel', 'wb') as f:
        pickle.dump(model, f)

def BiModelPredict():
    print("Model will be tested w/: {} logs.".format(len(testingVectors)))

    # Predicts remaining logSet data
    predictions = model.predict(testingVectors)

    # Prints calculated scores
    print("Accuracy score of trained binomial model: " + str(accuracy_score(testingLabels,predictions)))
    print("f1 score of trained binomial model: " + str(f1_score(testingLabels,predictions)))
    print("precision score of trained binomial model: " + str(precision_score(testingLabels,predictions)))
    print("recall score of trained binomial model: " + str(recall_score(testingLabels,predictions)))

    # Prints predictions that didn't match the corresponding labels, along with the log message.
    for i in range(len(testingVectors)):
        if predictions[i] != testingLabels[i]:
            print(["Line {}".format(i+1), predictions[i], testingLabels[i], logs[i]])

def MuModelPredict():
    print("Model will be tested w/: {} logs.".format(len(testingVectors)))

    # Predicts remaining logSet data
    predictions = model.predict(testingVectors)

    # Prints calculated scores
    print("Accuracy score of trained multinomial model: " + str(accuracy_score(testingLabels,predictions)))
    print("f1 score of trained multinomial model: " + str(f1_score(testingLabels,predictions)))
    print("precision score of trained multinomial model: " + str(precision_score(testingLabels,predictions)))
    print("recall score of trained multinomial model: " + str(recall_score(testingLabels,predictions)))

    # Prints predictions that didn't match the corresponding labels, along with the log message.
    for i in range(len(testingVectors)):
        if predictions[i] != testingLabels[i]:
            print(["Line {}".format(i+1), predictions[i], testingLabels[i], logs[i]])
