import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tkinter import Tk     # from tkinter import Tk for Python 3.x
import tkinter.filedialog as fd
import pickle
import time
import threading

# Bad actor labels for multinomial label conversion.
# Multinomial labels relate to index of respective bad actor assessment
classifications = ["safe", "ssh", "ws", "sql", "ddos", "ps", "su", "unsafe"]

# Class for creating threads
class myThread (threading.Thread):
    # Each thread initializes with a threadID number and an array of log files
    def __init__(self, threadID, logFiles):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.logFiles = logFiles
    
    # When the thread starts, it will announce itself, process the log files assigned to it, and will announce its completion.
    def run(self):
        print (self.name + " is starting.")
        analyzeLogs(self.logFiles)
        print (self.name + " completed.")

# Grabs and returns an array of user-selected filenames
# The files are chosen via a convenient File Explorer-like popup
def grabFilenames():
    # Reduces Tk() root functionality to allow basic functionality without extra features.
    Tk().withdraw()

    # Asks for filenames of the log files to be analyzed
    logFiles = fd.askopenfilenames(title='Please Select log file(s) to be analyzed.')

    return logFiles

def loadModels():
    global SVMBiModel, SVMMuModel, vectorizer

    # Asks for filename of Binomial SVM Algorithm and loads it into memory
    #filename = fd.askopenfilename(title='Please Select Binomial SVM Algorithm')
    with open('SVMBinomial.sav', 'rb') as f:
        SVMBiModel = pickle.load(f)

    # Asks for filename of Multinomial SVM Algorithm and loads it into memory
    #filename = fd.askopenfilename(title='Please Select Multinomial SVM Algorithm')
    with open('SVMMultinomial.sav', 'rb') as f:
        SVMMuModel = pickle.load(f)

    # Asks for filename of TfidfVectorizer and loads it into memory
    #filename = fd.askopenfilename(title='Please Select Vectorizer')
    with open('universalVectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

def createThreads():

    # Grabs filenames of log files to be processed from user
    # and loads ML algorithms
    logFiles = grabFilenames()
    loadModels()

    start = time.time()

    logFileNum = len(logFiles)

    # If more than 1 log file is selected by the user, two simultaneous threads are used
    # and the selected log files are split between them
    if logFileNum > 1:
        logFilesLists = np.array_split(np.array(logFiles), 2)
        thread1 = myThread(1, logFilesLists[0])
        thread2 = myThread(2, logFilesLists[1])

        # Starts threads
        thread1.start()
        thread2.start()

        # Waits for both threads to join before main continues
        thread1.join()
        thread2.join()
    else:
        analyzeLogs(logFiles)

    # Prints total time to analyze files
    print('Total time elapsed: {}s'.format(round(time.time()-start, 3)))


# Processes an array of log files and outputs predictions and stats to a text file.
def analyzeLogs(logFiles):
    
    # Analyze selection of log files given by user
    for logFile in logFiles:

        # Removes path from the log file name
        filename = (logFile.split('/')[-1])

        # Opens testing logs file and predicts one log message at a time
        with open("{}_predictions.txt".format(filename.split('.')[0]), "w") as predictionsFile: #"w" will overwrite any existing content, "a" will append to the end of the file. Will make a file called "AlgorithmOutput.txt" if one doesn't already exist
            # Starting time for analyzing current file
            start = time.time()

            predictionsFile.write('Starting predictions: {}\n\n'.format(filename))

            # Frequency dictionary for log message classifications
            frequencies = {'safe': 0, 'ws': 0, 'sql': 0, 'ddos': 0, 'ps': 0, 'ssh': 0, 'su': 0}

            # Counter for the next line to be processed
            lineNumber = 1

            with open(logFile, 'r', encoding='utf8') as logFile:
                
                while(logFile.readable()):
                    
                    # Batches of log messages are read in from the log file
                    # This value can be changed according to memory availability
                    logs = []
                    line = ''
                    for i in range(2000):
                        line = logFile.readline()
                        if line:
                            logs.append(line)
                        else:
                            break
                    
                    # If no logs were read from the log file, then log processing stops
                    if len(logs) == 0:
                        break

                    # Transforms log messages into vectors using TfidfVectorizer model
                    vector = vectorizer.transform(logs)
                    dense = vector.todense()
                    vectors = np.asarray(dense)

                    # Binomial SVM model predicts current log message
                    predictions = SVMBiModel.predict(vectors)

                    # k-means model will classify a log message if the binomial prediction was 'unsafe' i.e. the log message was a bad actor
                    for i in range(len(predictions)):
                        if predictions[i] == 'unsafe':
                            predictions[i] = SVMMuModel.predict(vectors[i].reshape(1,-1))[0]

                        # frequencies dictionary is updated with current prediction
                        frequencies.update({predictions[i]: frequencies.get(predictions[i]) + 1})
                    
                        # Writes prediction information to output text file
                        # Writes prediction, the log message, a regex-powered extracted features, and line number.
                        predictionsFile.write('Prediction: {}'.format(predictions[i]))
                        predictionsFile.write("\nlog: {}".format(logs[i])) # Write original log to file (not inluding '\n')
                        predictionsFile.write(featureExtract(logs[i]))
                        predictionsFile.write("\tLine Number: {}\n\n".format(lineNumber))

                        lineNumber += 1
            
            # Subtracts 1 to account for additional line added
            lineNumber -= 1

            # Records total time for analyzing current log file
            totalTime = round(time.time() - start, 3)

            #print(frequencies)
            # Writes frequencies of classifications to the output text file
            predictionsFile.write('Final Analysis: {}'.format(filename))
            predictionsFile.write('\nTotal Time for Analysis: {}s'.format(totalTime))
            predictionsFile.write('\nClassification Frequencies:')
            
            # Writes frequencies of safe and specific bad actor logs to output text file
            for badActor in frequencies.keys():
                predictionsFile.write("\n\t{}: {:.2%}".format(badActor, frequencies.get(badActor)/lineNumber))

            # Frequency of unsafe logs in the 1 - the frequency of safe logs
            predictionsFile.write("\n\t{}: {:.2%}".format('unsafe', 1-(frequencies.get('safe')/lineNumber)))


# Uses regex to extract features from logs and returns the features in a convenient format.
def featureExtract(line):
    features = ""
    if re.search("^\[",line): #log starts with a square bracket
       # print("This is a error.log log file")
        line = re.split("\[", line, 1)[1]
        #date = re.split("(?<=.{10})\s", line, 1)[0] #gets first part of date. Will be concatenated with year later
        #line = re.split("(?<=.{10})\s", line, 1)[1]
        [date, line] = re.split("(?<=.{10})\s", line, 1)[0:2]
        [time, line] = re.split("\s", line, 1)[0:2]
        #time = re.split("\s", line, 1)[0]
        #line = re.split("\s", line, 1)[1]
        date = date + " " + re.split("]", line, 1)[0] #concatenates year with the rest of the date
        line = re.split("]", line, 1)[1]
        line = re.split("\[", line, 1) [1]
        logType = re.split("\]", line, 1) [0]
        line = re.split("\[", line, 1) [1]
        IDs = re.split("\]", line, 1) [0]
        PID = re.split("\s", IDs, 1)[1]
        PID = re.split("\:", PID, 1)[0]
        TID = re.split("\s", IDs, 2)[2]
        client = ""
        if re.search("\[(?=client)", line): #ignore if brackets arent for client
            line = re.split("\[", line, 1)[1]
            line = re.split("\s", line, 1)[1]
            client = re.split("\]", line, 1)[0]
        line = re.split("\s(?=A)", line, 1)[1] #find space followed by "A" 
        errorCode = re.split("\:", line, 1)[0]
        msg = re.split("\:", line, 1)[1]
        features = "\ttime = " + time + "\n\t" + "date = " + date + "\n\t" + "log type = " + logType + "\n\t" + "PID = " + PID + "\n\t" + "TID = " + TID + "\n\t" + "client = " + client + "\n\t" + "error code = " + errorCode + "\n\t" + "message =" + msg
        
    elif re.search("^\d",line): #log starts with a digit
        #print("This is a access.log log file") #used for malicious web server access bad actor
        #log format:
        #ip - user (- if not relevant) [date & time] "GET \X"-"(if not needed)
            #else "GET \X" \d \d "http://ip/" "Browser information"
        #ip = re.split("\s", line, 1)[0]
        #line = re.split("\s", line, 1)[1]
        [ip, line] = re.split("\s", line, 1)[0:2]
        line = re.split("(?<=-)\s", line, 1)[1]
        user = re.split("\s", line, 1)[0]
        line = re.split("\s\[", line, 1)[1]
        #date = re.split(":", line, 1)[0]
        #line = re.split(":", line, 1)[1]
        [date, line] = re.split(":", line, 1)[0:2]
        time = re.split("]", line, 1)[0]
        msg = re.split("] ", line, 1)[1]
        features = "\tip = " + ip + "\n\t" + "user = " + user + "\n\t" + "date = " + date + "\n\t" + "time = " + time + "\n\t" + "message = " + msg

    elif re.search("^[A-Za-z]",line): #log starts with a alphabetical character
        #print("This is a auth.log log file")
        #log format:
        #month  day hour:minute:second user useraccount logtype: main information
        #date = re.split("(?<=.{6})\s", line, 1)[0]
        #line = re.split("(?<=.{6})\s", line, 1)[1]
        [date, line] = re.split("(?<=.{6})\s", line, 1)[0:2]
        #time = re.split("(?<=.{8})\s", line, 1)[0]
        #line = re.split("(?<=.{8})\s", line, 1)[1]
        [time, line] = re.split("(?<=.{8})\s", line, 1)[0:2]
        user = re.split("\s", line)[0]
        msg = re.split("\s", line, 1)[1]
        features = "\tdate = " + date + "\n\t" + "time = " + time + "\n\t" + "user = " + user + "\n\t" + "message = " + msg
    return features

createThreads()
