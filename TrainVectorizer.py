import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import Button
from tkinter import messagebox
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter import filedialog as fd

trainVectorizerRoot = Tk()

# Creates UI for trainVectorizer program
def trainVectorizerUI():

    # UI buttons for user to press
    Button(trainVectorizerRoot, text='Train TfidfVectorizer', 
        command=verifyTrainInput).pack(fill=tk.X)

    Button(trainVectorizerRoot, text='Exit', 
        command=trainVectorizerRoot.destroy).pack(fill=tk.X)

    trainVectorizerRoot.mainloop()

# Verifies selection to train a new TfidfVectorizer model
def verifyTrainInput():
    ans = messagebox.askquestion(title="Are you sure?", message="This will overwrite any existing TfidfVectorizer model in the current directory.")
    if ans == 'yes':
        trainVectorizer()

# Creates, fits, and exports TfidfVectorizer model with an user-provided log file
def trainVectorizer():

    # Asks user for log file to fit vectorizer model with
    FitLogsFilename = fd.askopenfilename(title='Please select log file for fitting model')

    with open(FitLogsFilename, 'r', encoding='utf8') as fitLogs:

        # Creates and fit TfidfVectorizer model
        vectorizer = TfidfVectorizer()
        vectorizer.fit(fitLogs)
        print("Vectorizer Trained.")

        # Exports it as a .pickle file using pickle.dump()
        pickle.dump(vectorizer, open('universalVectorizer.pickle', 'wb'))
        
        messagebox.showinfo(message='TfidfVectorizer Created.')

trainVectorizerUI()