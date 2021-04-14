from os import error, write
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import json
import numpy as np
import re
import random

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

with open(filename, 'r', encoding='utf8') as logs:
    errorLogs = open('errorLogs.txt', 'w', encoding='utf8')
    authLogs = open('authLogs.txt', 'w', encoding='utf8')
    accessLogs = open('accessLogs.txt', 'w', encoding='utf8')
    for line in logs:
        if re.search("^\[",line): #log starts with a square bracket, indicating this is an error.log log
            errorLogs.write(line)
        elif re.search("^\d",line): #log starts with a digit, indicating this is an access.log log used for malicious web server access bad actor
            accessLogs.write(line)
        elif re.search("^[A-Za-z]",line): #log starts with a alphabetical character, indicating this is an auth.log log
            authLogs.write(line)

    errorLogs.close()
    accessLogs.close()
    authLogs.close()

    with open('organizedLogs.txt', 'w', encoding='utf8') as organizedLogs:

        badActorLabels = ('safe','ssh','ws','sql','ddos','ps','su')
        biLabels = []
        muLabels = []
        biIndex = 0
        muIndex = 0
        badActorBiLabels = []
        badActorMuLabels = []

        safeLogs = open('safeLogs.txt', 'w', encoding='utf8')
        sshLogs = open('sshlogs.txt', 'w', encoding='utf8')
        wsLogs = open('wsLogs.txt', 'w', encoding='utf8')
        sqlLogs = open('sqlLogs.txt', 'w', encoding='utf8')
        ddosLogs = open('ddosLogs.txt', 'w', encoding='utf8')
        psLogs = open('psLogs.txt', 'w', encoding='utf8')
        suLogs = open('sulogs.txt', 'w', encoding='utf8')
        badActorLogs = open('badActorLogs.txt', 'w', encoding='utf8')

        organizedLogs.write('error.log\n')
        with open('errorLogs.txt', 'r', encoding='utf8') as errorLogs:
            for line in errorLogs:
                if 'AH01618' in line:
                    wsLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('ws'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                elif 'AH01617' in line:
                    sqlLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('sql'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                else:
                    safeLogs.write(line)
                    biLabels.append(0)
                    muLabels.append(badActorLabels.index('safe'))
                organizedLogs.write(line)
        #organizedLogs.write(json.dumps(biLabels[biIndex: ]))
        #organizedLogs.write(json.dumps(muLabels[muIndex: ]))
        #organizedLogs.write('\n')
        #biIndex = len(biLabels)
        #muIndex = len(muLabels)

        organizedLogs.write('\nauth.log\n')
        with open('authLogs.txt', 'r', encoding='utf8') as authLogs:
            for line in authLogs:
                if 'sshd[' in line and ('authentication failure' in line or 'maximum authentication attempts' in line):
                    sshLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('ssh'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                elif 'sshd[' in line and 'version differ' in line:
                    psLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('ps'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                elif 'sshd[' in line and 'no matching' in line:
                    psLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('ps'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                # 'sudo apt autoremove' and 'sudo apt-get autoremove' are flagged as a bad actor.
                elif 'sudo' in line and 'remove' in line:
                    suLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('su'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                # Failed password attempts are flagged as bad actors
                elif 'sudo' in line and 'incorrect password attempts' in line:
                    suLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('su'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                elif 'sudo' in line and 'authentication failure' in line:
                    suLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('su'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                else:
                    safeLogs.write(line)
                    biLabels.append(0)
                    muLabels.append(badActorLabels.index('safe'))
                organizedLogs.write(line)
        #organizedLogs.write(json.dumps(biLabels[biIndex: ]))
        #organizedLogs.write(json.dumps(muLabels[muIndex: ]))
        #organizedLogs.write('\n')
        #biIndex = len(biLabels)
        #muIndex = len(muLabels)

        organizedLogs.write('\naccess.log\n')
        with open('accessLogs.txt', 'r', encoding='utf8') as accessLogs:
            for line in accessLogs:
                if 'ApacheBench/2.3' in line:
                    ddosLogs.write(line)
                    badActorLogs.write(line)
                    biLabels.append(1)
                    muLabels.append(badActorLabels.index('ddos'))
                    badActorBiLabels.append(biLabels[-1])
                    badActorMuLabels.append(muLabels[-1])
                else:
                    safeLogs.write(line)
                    biLabels.append(0)
                    muLabels.append(badActorLabels.index('safe'))
                organizedLogs.write(line)
        #organizedLogs.write(json.dumps(biLabels[biIndex: ]))
        #organizedLogs.write(json.dumps(muLabels[muIndex: ]))
        #organizedLogs.write('\n')
        #biIndex = len(biLabels)
        #muIndex = len(muLabels)

        safeLogs.close()
        sshLogs.close()
        wsLogs.close()
        sqlLogs.close()
        ddosLogs.close()
        psLogs.close()
        suLogs.close()
        badActorLogs.close()

        with open('organizedLogs_Bi_labels.txt', 'w', encoding='utf8') as biLabelFile:
            json.dump(biLabels, biLabelFile)
        with open('organizedLogs_Mu_labels.txt', 'w', encoding='utf8') as muLabelFile:
            json.dump(badActorLabels, muLabelFile)
            json.dump(muLabels, muLabelFile)
        
        with open('badActors_Bi_labels.txt', 'w', encoding='utf8') as biLabelFile2:
            json.dump(badActorBiLabels, biLabelFile2)
        with open('badActors_Mu_labels.txt', 'w', encoding='utf8') as muLabelFile2:
            json.dump(badActorMuLabels, muLabelFile2)
        
        total = 1000
        totalLeft = total
        badActorPerc = 25
        universalBiLabels = []
        universalMuLabels = []
        universalSet = []

        print('Size of Universal Set: {}'.format(total))
        with open('universalSet.txt', 'w', encoding='utf8') as set:
            for filename in badActorLabels[1:]:
                with open('{}Logs.txt'.format(filename), 'r', encoding='utf8') as logs:
                    lines = logs.readlines()
                    random.shuffle(lines) 
                    for i in range( min( len(lines), int((total*(badActorPerc/100))//( len(badActorLabels[1:] ))) ) ):
                        universalSet.append(lines[i])
                        totalLeft -= 1
                        universalBiLabels.append(1)
                        universalMuLabels.append(badActorLabels.index(filename))
            print('Number of Unique Bad Actors: {}'.format(len(badActorLabels[1:] )))
            print('Expected Percentage of Bad Actors: {}%'.format(badActorPerc))
            print('Bad Actors in Universal Set: {}'.format(total - totalLeft))

            sshSafeLimit = int((total*(1-(badActorPerc/100)))//( len(badActorLabels[1:] )))
            sshEveryOther = True
            with open('safeLogs.txt','r', encoding='utf8') as source:
                lines = source.readlines()
                random.shuffle(lines)
                for line in lines:
                    if totalLeft > 0:
                        if 'sshd[' in line and sshEveryOther and sshSafeLimit > 0:
                            universalSet.append(line)
                            sshEveryOther = not sshEveryOther
                            sshSafeLimit -= 1
                            universalBiLabels.append(0)
                            universalMuLabels.append(badActorLabels.index('safe'))
                            totalLeft -= 1
                        elif 'sshd[' in line and not sshEveryOther and sshSafeLimit > 0:
                            sshEveryOther = not sshEveryOther
                        else:
                            universalSet.append(line)
                            universalBiLabels.append(0)
                            universalMuLabels.append(badActorLabels.index('safe'))
                            totalLeft -= 1

            logs = np.array(universalSet)
            biLabels = np.array(universalBiLabels)
            muLabels = np.array(universalMuLabels)

            randomize = np.arange(len(logs))
            np.random.shuffle(randomize)
            logs = logs[randomize]
            biLabels = list(biLabels[randomize])
            muLabels = list(muLabels[randomize])
            for i in range(len(biLabels)):
                biLabels[i] = biLabels[i].item()
                muLabels[i] = muLabels[i].item()

            with open('universalSet_biLabels.txt', 'w', encoding='utf8') as biLabelFile:
                with open('universalSet_muLabels.txt', 'w', encoding='utf8') as muLabelFile:
                    for line in logs:
                        set.write(line)
                    json.dump(biLabels, biLabelFile)
                    json.dump(muLabels, muLabelFile)
print('\nLog file is sorted, and universal set is created.')