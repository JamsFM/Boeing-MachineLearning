import SVM
import program
import TrainVectorizer
import tkinter as tk
from tkinter import Tk
from tkinter import messagebox


def SelectFile():
    global root
    root = Tk()
    DestroyNonRoot()

    
    tk.Button(root, text='Predict New Log Files', 
        command=program.createThreads).pack(fill=tk.X)

    tk.Button(root, text='Training Models and Scoring Predictions', 
        command=SVMSelections).pack(fill=tk.X)

    tk.Button(root, text='Create Vectorizer Model',
        command=VerifyVectorizer).pack(fill=tk.X)

    root.mainloop()

def SVMSelections():
    global svm
    svm = Tk()
    root.destroy()
    tk.Button(svm, text='Train Binomial Model', 
        command=VerifyBinomial).pack(fill=tk.X)

    tk.Button(svm, text='Train Multinomial Model', 
        command=SVM.MultinomialModelTrain).pack(fill=tk.X)

    tk.Button(svm, text='Score Predictions Using Pipeline', 
        command=lambda: SVM.MakePredictions(True)).pack(fill=tk.X)

    tk.Button(svm, text='Return to Previous Menu', 
        command=SelectFile).pack(fill=tk.X)

    svm.mainloop()

def VerifyVectorizer():
    ans = messagebox.askquestion(title="Are you sure?", message="This will overwrite any existing TfidfVectorizer model in the current directory.")
    if ans == 'yes':
        TrainVectorizer.trainVectorizer()

def VerifyBinomial():
     ans = messagebox.askquestion(title="Are you sure?", message="This will overwrite any existing Binomial SVM model in the current directory.")
     if ans == 'yes':
        SVM.BinomialModelTrain()

def VerifyMultinomial():
     ans = messagebox.askquestion(title="Are you sure?", message="This will overwrite any existing Multinomial SVM model in the current directory.")
     if ans == 'yes':
        SVM.MultinomialModelTrain()



def DestroyNonRoot():
    try:
        svm
    except:
        print(end = "")
    else:
        svm.destroy()

SelectFile()

