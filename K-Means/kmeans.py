import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tkinter import Tk 
import tkinter.filedialog as fd
import pickle

# Opens log files & makes k-means model and writes predictions to cluster files
def kmeans(numClusters):

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

    # Grabs the filenames and paths of the training/fitting logs, testing logs, and vectorizer
    trainingLogsFile = fd.askopenfilename(title='Select Training Logs File')
    testingLogsFile = fd.askopenfilename(title='Select Testing Logs File')
    vectorizerFilename = fd.askopenfilename(title='Select Vectorizer Mode File')

    # Opens and loads vectorizer model
    with open(vectorizerFilename, 'rb') as f:
        vectorizer = pickle.load(f)

    # Opens and transforms training/fitting logs into vectors using vectorizer
    with open(trainingLogsFile, 'r', encoding='utf8') as f:
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            trainingVectors = np.asarray(dense)

    # Opens and transforms testing logs into vectors using vectorizer
    with open(testingLogsFile, 'r', encoding='utf8') as f:
            vectors = vectorizer.transform(f)
            dense = vectors.todense()
            testingVectors = np.asarray(dense)

            # Stores testing logs in memory for writing to cluster text files
            f.seek(0)
            logs = []
            for message in f:
                logs.append(message)

    # Creates k-means model with user-specified clusters
    model = KMeans(n_clusters=numClusters)

    # Fits model
    model.fit(trainingVectors)

    # Saves model using pickle.dump()
    with open('kmeans.pickel', 'wb') as f:
        pickle.dump(model, f)

    # Generates an array of integers that tells which cluster the k-means algorithm
    # sorted a testing log into
    predicted = model.predict(testingVectors)

    # Grabs the current working directory
    path = os.getcwd()

    # Generates a text file for each cluster and writes corresponding logs to
    # the text file. Creates a new folder for the text files
    for i in range(numClusters):

        # Combines current working directory with a new folder and current cluster's text file name
        filename = "{}/KmeansClusters/cluster{}.txt".format(path, i)

        # Creates "KmeansCluster" folder if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Goes through the prediction array and writes any logs that got a prediction
        # that matches the current cluster
        with open(filename, 'w', encoding='utf8') as file:
            for j in range(len(predicted)):
                if predicted[j] == i:
                    file.write(logs[j])

    # Confirmation message
    print("done")