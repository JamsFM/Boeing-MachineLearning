import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tkinter import Tk     # from tkinter import Tk for Python 3.x
import tkinter.filedialog as fd



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
        


