import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import metrics
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import math
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import json
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#from sklearn.metrics import f1_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score


Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
fileNames = []
for i in range(5): #in order: trainingSet,testingSet,trainingLabels,testingLabels,vectorizerModel
     fileNames.append(askopenfilename())

global trainingLogs, testingLogs, trainingLabels, testingLabels
with open(fileNames[4], 'rb') as f:
       vectorizer = pickle.load(f)

with open(fileNames[0], 'r', encoding='utf8') as f:
      vectors = vectorizer.transform(f)
      dense = vectors.todense()
      trainingLogs = np.asarray(dense)

with open(fileNames[1], 'r', encoding='utf8') as f:
      vectors = vectorizer.transform(f)
      dense = vectors.todense()
      testingLogs = np.asarray(dense)

with open(fileNames[2], 'r', encoding='utf8') as f:
      trainingLabels = json.load(f)


with open(fileNames[3], 'r', encoding='utf8') as f:
      testingLabels = json.load(f)


testingLabels = np.array(testingLabels)
model = keras.Sequential([
                          keras.layers.Flatten(input_shape = (len(trainingLogs[1]),)), #array of unspecified length, with subarrays the length of trainingLogs
                          keras.layers.Dense(64, activation = "relu"), #Number of nodes in the hidden layer
                          keras.layers.Dense(2,activation="softmax") # number of possible labels
                          ])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"]) #found nadam and adam to be best optimizers

history = model.fit(trainingLogs, trainingLabels, epochs=10) #epochs is number of times it will run through and train on the data
testLoss, testAcc = model.evaluate(testingLogs, testingLabels)
prediction = model.predict(testingLogs)

predictions = []
for i in range (len(prediction)):
    predictions.append(np.argmax(prediction[i]))

"""multi
print("Accuracy score of trained multinomial model: " + str(accuracy_score(testingLabels,predictions)))
print("f1 score of trained multinomial model: " + str(f1_score(testingLabels,predictions,average='micro')))
print("precision score of trained multinomial model: " + str(precision_score(testingLabels,predictions,average='micro')))
print("recall score of trained multinomial model: " + str(recall_score(testingLabels,predictions,average='micro')))
"""

print("Accuracy score of trained binomial model: " + str(accuracy_score(testingLabels,predictions)))
print("f1 score of trained binomial model: " + str(f1_score(testingLabels,predictions,pos_label=0)))
print("precision score of trained binomial model: " + str(precision_score(testingLabels,predictions,pos_label=0)))
print("recall score of trained binomial model: " + str(recall_score(testingLabels,predictions,pos_label=0)))

"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy and Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss'], loc='best')
plt.show()
"""

#categories = ["Safe  ", "Unsafe"]

#for i in range (len(testingLabels)):
   #  print('Actual: ' , testingLabels[i], '  Prediction: ' , np.argmax(prediction[i]))
