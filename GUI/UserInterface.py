import SVM, program, TrainVectorizer, KMeans
import tkinter as tk
from tkinter import simpledialog, Tk, messagebox


def SelectAction():
    global root
    root = Tk()
    root.title("What would you like to do?")
    root.geometry("500x200") #Set window size
    
    tk.Button(root, text='Predict New Log Files', #Run main program
        command=program.createThreads).pack(fill=tk.X)

    tk.Button(root, text='Developer Tools', #Open new window with developer tools options
        command=DeveloperTools).pack(fill=tk.X)

    root.mainloop()

#Has developer options: Train new Binomial and MultiNomial Models,
#Score predictions, create new vectorizer, and cluster logs with K-means
def DeveloperTools():
    global dev
    dev = Tk()
    dev.title("Developer Tools")
    root.withdraw()
    dev.geometry("500x200") #Set window size
   
    tk.Button(dev, text='Train Binomial Model', 
        command=VerifyBinomial).pack(fill=tk.X)

    tk.Button(dev, text='Train Multinomial Model', 
        command=VerifyMultinomial).pack(fill=tk.X)

    tk.Button(dev, text='Score Predictions Using Pipeline', 
        command=lambda: SVM.MakePredictions(True)).pack(fill=tk.X)

    tk.Button(dev, text='Create Vectorizer Model',
        command=VerifyVectorizer).pack(fill=tk.X)

    tk.Button(dev, text='Create K-Means Clusters',
        command=KMeansCluster).pack(fill=tk.X)

    tk.Button(dev, text='Return to Previous Menu', #go back to main menu and close developer tools
        command=lambda: [dev.withdraw(), SelectAction()]).pack(fill=tk.X)

    dev.mainloop()

#Verify user wants to overwrite existing vectorizer model
def VerifyVectorizer():
    dev.withdraw()
    ans = messagebox.askquestion(title="Are you sure?", message="This will overwrite any existing TfidfVectorizer model in the current directory.")
    if ans == 'yes':
        TrainVectorizer.trainVectorizer()
        messagebox.showinfo(title = None, message = "Vectorizer Model Created")
    dev.deiconify()

#Verify user wants to overwrite existing Binomial model
def VerifyBinomial():
     dev.withdraw()
     ans = messagebox.askquestion(title="Are you sure?", message="This will overwrite any existing Binomial SVM model in the current directory.")
     if ans == 'yes':
        SVM.BinomialModelTrain()
        messagebox.showinfo(title = None, message = "Binomial Model Created")
     dev.deiconify()
    
#Verify user wants to overwrite existing Multinomial model
def VerifyMultinomial():
     dev.withdraw()
     ans = messagebox.askquestion(title="Are you sure?", message="This will overwrite any existing Multinomial SVM model in the current directory.")
     if ans == 'yes':
        SVM.MultinomialModelTrain()
        messagebox.showinfo(title = None, message = "Multinomial Model Created")
     dev.deiconify()

#Ask user how many clusters k-means analysis should have
def KMeansCluster():
    dev.withdraw()
    promptInput = Tk()
    tk.Label(promptInput, text="Please enter desired number of log clusters").pack()
    numClusters = tk.Entry(promptInput)
    numClusters.pack()

    tk.Button(promptInput, text='Create Clusters', 
        command=lambda: [KMeans.kmeans(int(numClusters.get())), promptInput.destroy(), DeveloperTools()]).pack(fill=tk.X)
    
    promptInput.mainloop()

 
SelectAction()

