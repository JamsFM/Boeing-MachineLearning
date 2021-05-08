# **Boeing-MachineLearning**
##### **Google Drive:** (Relevant Documents: Design Spec, Validation/Test Plan, PRD)
##### https://drive.google.com/drive/u/1/folders/0AF8Y3XQxwJHZUk9PVAhttps://drive.google.com/drive/u/1/folders/0AF8Y3XQxwJHZUk9PVA


### **Boeing Analytics for Plane and Ground Support Log Data**
#### Design Specifications 0.7
#### *Nicholas Falter, James Mare, Riley Ruckman, Travis Shields*
#### managed by: *Nicholas Falter*
#### 4/11/2021

Revision History:
This section lists the previous versions of this document. Details as to what changed between versions of this document can be found in this section.

Table 1. Each revision, and information about the revision.
Revision Number:
Date:
Revision Contents:
Author:
0.1
10/15/2020
Document structure
Nicholas Falter
0.2
11/5/2020
Early Project Description
Nicholas Falter
0.3
11/29/2020
Updated Project Description
Nicholas Falter
0.4
12/13/2020
Added machine learning pipeline, and information about bad actors.
Nicholas Falter
0.5
1/24/2021
Added detail about what MLP steps pertain to in our project - system architecture 
Added more detail, and divided into subsections - system requirements
Reorganized, added unauthorized web server logins bad actor, added more detail to DDOS bad actor - system design
Rewrote to be more detailed - ethical considerations
Nicholas Falter
0.6
2/20/2021
Updated system architecture to include bad actor detection pipeline.
Updated log parameterization
Nicholas Falter
0.7
4/11/2021
Added machine learning section to system design, updated description of machine learning throughout entire document.
Nicholas Falter

A table detailing all revisions of this document and information about the revisions

Table of Contents:
This section lists all of the different sections in this document. Page numbers can be found for different sections.

1. Overview/Introduction	1
2. Requirements	2
2.1 Highest Priority/Primary Project Goal	2
2.2 Program Constraints	2
3. System Architecture	3
3.1 Raw Data	4
3.2 Data Preprocessing	4
3.3 Model Training	5
3.4 Model Validation	5
3.5 Deploy for Serving/Serving Endpoint	5
4. System Design	6
4.1 Bad Actor Scenarios	6
4.1.a Unauthorized Use of Root/Superuser Privileges	6
4.1.b Malicious Web Server Access	7
4.1.c Unauthorized SSH Access	8
4.1.d Unauthorized Web Server Logins	9
4.1.e Port Scan	10
4.1.f DDoS	11
4.2 Log Processing	12
4.3 Machine Learning	13
4.3.a Neural Networks	14
4.3.b Decision Trees	14
4.3.c Naive Bayes Classifier	14
4.3.d Support Vector Machines (SVM)	14
4.3.e K-Means	15
4.4 Human Interface Design	15
5. Project Bill of Materials (BOM)	16
6. Ethical Considerations	17
7. References	18
8. Errata	19


1. Overview/Introduction
This section gives a general overview of what this project is. As we progress this will be updated to explain what it is we are working on and what testing will be covered in this document. 

The purpose of this project is to create a proof of concept program that simulates analyzing log data from aircraft and ground support systems to discover and report anomalies in this log data. This would tell signs of potential threats to the aircraft.
The scope of this project does not include analyzing actual log data from aircraft or ground support systems. Cyber threats are simulated in a Linux Ubuntu environment, and the logs generated from these simulated threats are what is required to be detected by the program. Logs from real aviation systems would be required to create a program that is efficient at detecting threats in an aviation environment. If Boeing is satisfied with the performance of the proof of concept program created in this project, they may create a similar program using the same methodology, with real aviation logs that could be implemented in a real aviation environment.
This project also includes creating a recommended set of procedures for analyzing these logs, with considerations such as the cost-effectiveness and probability of detecting a potential threat. The best methods for airports will vary, so potential considerations for an airport to examine will be outlined, but no universal best method can be determined.
Currently, there are 6 different bad actors being used as data for the machine learning algorithm. The bad actors are unauthorized use of root/superuser privileges, malicious web server access, unauthorized logins, unauthorized web server logins, port scans, and DDOS. This is a very small subset of the bad actors Boeing tests for, but for the purposes of this project, it is adequate.

2. Requirements
This section describes the critical requirements this project aims to satisfy.  This section summarizes what is documented in the PRD.
2.1 Highest Priority/Primary Project Goal
Create a program with machine learning that is able to identify potential threats to a computer’s cyber security by analyzing the computer’s logs. Logs are generated on a Linux system and these logs are what are required to be identified. This program simulates analyzing an aircraft system logs, but will not be tested or programmed with actual aircraft logs. This must be implemented through machine learning to be able to identify threats that were not explicitly programmed for or tested. 
2.2 Program Constraints
Boeing has given us freedom to choose the program constraints. Our chosen program constraints are to make the program capable of reading 50 GB of log data at a time to handle the large amount of log data that is generated between analysis in an aircraft environment. The amount of logs generated is variable, and the testing period can be changed. One aircraft can generate up to 250+MB of logs per day, which is why this value was chosen. It needs to be able to analyze this amount of log data in a maximum of two days. It also needs to be able to correctly identify bad actors around 80% of the time, and it needs to be able to run on Windows.
Boeing requires the dataset of bad actors to have at least five entries. We chose to create the dataset from: unauthorized use of root/superuser privileges, malicious web server access, unauthorized logins, port scans, and DDOS. Additionally, unauthorized web server logins have been added to the dataset past the minimum of five.
We have the freedom to select any machine learning algorithm we see fit. We tested Naive Bayes, Kmeans, Decision Trees, Neural Networks, Support Vector Machines (SVM), and Logistic Regression.

3. System Architecture
This section describes at a high level how this project works. It also describes alternate methods that could be used and why our team decided to use the methods we did.

Transmitting logs wirelessly prevents personnel from having to physically visit each log producing system, which saves man-hours. If the airport doesn’t have WiFi, then this isn’t an option, but an analysis of how many man-hours are spent on collecting logs, and how much it would cost to install and maintain the internet connection, could be performed to determine if the airport should be set up to wirelessly transmit logs. Log analysis should be performed the day before an aircraft’s flight, and additionally be performed every week to ensure the integrity of the aircraft.
This project implements a machine learning algorithm, support vector machines (SVM) to identify and contextualize logs that potentially indicate a threat to an airplane.
Using machine learning to create a program looks as follows:

Figure 1. Steps involved in creating a program using machine learning.


There are several different methods we can use to identify bad actors. We can use supervised learning to binomially identify whether a log is a bad actor (potential cybersecurity threat) or not. After we have identified if it is a bad actor or not, we can further identify which bad actor it is. We can also use unsupervised learning to identify new bad actors that our model isn’t trained to detect yet.





Figure 2. Bad actor detection methods.

A more detailed explanation of the machine learning steps can be found in Section 4.3.
3.1 Raw Data
The raw data is the logs generated from simulating bad actors (potential cybersecurity threats) within a Linux Ubuntu environment. The bad actors examined so far are: unauthorized use of root/superuser privileges, malicious web server access, unauthorized logins, port scans, DDOS, and unauthorized web server logins.

3.2 Data Preprocessing
The logs from these bad actor scenarios are then parameterized and converted into a numerical form. This is done using a model that gives weight relative to how frequent a word is in a log and removes weight relative to how frequent that word is in all the logs that have been processed. The numerical data is then given to the machine learning algorithm where it is then used to train and test the model.
3.3 Model Training
We will use the machine learning algorithm Support Vector Machines (SVM) trained by the dataset we constructed of cybersecurity risks to create a program capable of detecting logs indicating a cybersecurity risk. This algorithm was chosen because it was the most accurate of all algorithms tested.
3.4 Model Validation
	The resulting program will be tested upon a set of log data, where its ability to detect bad actors will be determined. If the resulting program meets the requirements outlined in Section 2.2, then the model is satisfactory. The best performing model is selected.
3.5 Deploy for Serving/Serving Endpoint
	This section of the project will not be handled by us. Boeing will review the results of our projects and if they are satisfied with the results, they will recreate the project with authentic airplane logs, and then use their recreation of our program to serve their endpoint(s), which would be analyzing logs from aircraft and their subsystems, and detecting logs that indicate bad actors (potential threats to the airplane’s cybersecurity).

4. System Design
This section gives specific details as to how each component of this project works. This section is useful for replicating similar projects.

We are using Linux system logs to simulate airplane logs. These logs are then used with the machine learning algorithm Support Vector Machines (SVM) to identify and contextualize bad actors (logs that potentially indicate a threat to an airplane). These bad actors are simulated in a Linux virtual machine. The logs generated are then combined into a dataset, which is processed into a numeric form that can be used to train a machine learning algorithm.
4.1 Bad Actor Scenarios
Bad actor scenarios are cyber-security events that potentially pose a threat to the aircraft. The following are the procedures and results we have for the bad actors used in this project.
4.1.a Unauthorized Use of Root/Superuser Privileges
Logs: 
// Failed Login //
Dec  9 19:03:03 riley-VirtualBox sudo: pam_unix(sudo:auth): authentication failure; logname= uid=1000 euid=0 tty=/dev/pts/1 ruser=riley rhost=  user=riley

// Successful Login //
Dec  9 19:03:20 riley-VirtualBox sudo:    riley : TTY=pts/1 ; PWD=/home/riley/Desktop ; USER=root ; COMMAND=/usr/bin/apt update
Dec  9 19:03:20 riley-VirtualBox sudo: pam_unix(sudo:session): session opened for user root by (uid=0)
Dec  9 19:03:21 riley-VirtualBox sudo: pam_unix(sudo:session): session closed for user root

Procedure: 
Access a command terminal and perform a command with the keyword “sudo” stated first. This will prompt for a password to be entered. If successful, the command will continue on as normal. If not successful, an error will appear along with another try for password entry. When the sudo keyword is used successfully, the log message will include the user, some other information, and the specific command that was called. Certain sudo commands are marked as unsafe and others are safe. Failing to enter the correct password for sudo privileges also falls under this category. Logs are at: /var/log/auth.log.
4.1.b Malicious Web Server Access
Setup: How To Install the Apache Web Server on Ubuntu 18.04
How To Set Up Password Authentication with Apache on Ubuntu 16.04

Logs:
[Sun Dec 06 15:27:53.417366 2020] [auth_basic:error] [pid 8235:tid 139932450420480] [client 166.181.249.13:49766] AH01617: user admin: authentication failure for “/”: Password Mismatch
[Sun Dec 06 15:30:42.481279 2020] [auth_basic:error] [pid 8235:tid 139933717096192] [client 166.181.249.13:28554] AH01617: user admin: authentication failure for “/”: Password Mismatch
[Sun Dec 06 15:32:42.415483 2020] [auth_basic:error] [pid 8235:tid 139933415089920] [client 166.181.249.13:6858] AH01617: user admin: authentication failure for “/”: Password Mismatch
[Sun Dec 06 15:33:08.543591 2020] [auth_basic:error] [pid 8234:tid 139932895004416] [client 166.181.249.13:65234] AH01617: user admin: authentication failure for “/”: Password Mismatch
[Sun Dec 06 15:33:27.798207 2020] [auth_basic:error] [pid 8234:tid 139932777572096] [client 166.181.249.13:9627] AH01617: user admin: authentication failure for “/”: Password Mismatch
[Sun Dec 06 15:34:13.093943 2020] [auth_basic:error] [pid 8234:tid 139933683525376] [client 166.181.249.13:54685] AH01617: user admin: authentication failure for “/”: Password Mismatch
[Sun Dec 06 15:34:33.137435 2020] [auth_basic:error] [pid 8235:tid 139933071152896] [client 166.181.249.13:37470] AH01617: user admin: authentication failure for “/”: Password Mismatch

Procedure: 
Access website, and when prompted to enter username and password, enter username and enter a command into the password field. Example commands here:
https://www.youtube.com/watch?v=ciNHn38EyRc
Since the logs do not show what the failed password is, the logs for entering an invalid password are the same for entering an SQL command into the password field.
Logs are at: /var/log/apache2/error.log


Screenshot:
Figure 3. Malicious Web Server Access Logs - Screenshot

4.1.c Unauthorized SSH Access
Logs:
Dec 11 18:59:11 Falter-VirtualBox gdm-password]: pam_unix(gdm-password:auth): Couldn't open /etc/securetty: No such file or directory
Dec 11 18:59:11 Falter-VirtualBox gdm-password]: pam_unix(gdm-password:auth): Couldn't open /etc/securetty: No such file or directory
Dec 11 18:59:11 Falter-VirtualBox gdm-password]: pam_unix(gdm-password:auth): authentication failure; logname= uid=0 euid=0 tty=/dev/tty1 ruser= rhost=  user=falter
1585 Dec  5 18:15:87 travis sshd[11070]: authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=174.214.13.29  user=travis
1586 Dec  5 18:15:89 travis sshd[11070]: Failed password for travis from 174.214.13.29 port 14191 ssh2
1587 Dec  5 18:15:14 travis sshd[11070]: Accepted password for travis from 174.214.13.29 port 14191 ssh2
1588 Dec  5 18:15:14 travis sshd[11070]: pam_unix(sshd:session): session opened for user travis by (uid=0)
1589 Dec  5 18:15:14 travis systemd-logind[722]: New session 21 of user travis.

Procedure: 
To simulate this event, we allowed port forwarding to SSH port 22, and failed a few logins before successfully logging in. For both cases, a single failed login does not indicate a bad actor, because it is possible a valid user failed to enter their password correctly. After 3 failed logins, it should be considered a bad actor. Logs are at: /var/log/auth.log.

Screenshot: 
Figure 4. Unauthorized Logins Logs - Screenshot


4.1.d Unauthorized Web Server Logins
Setup: How To Install the Apache Web Server on Ubuntu 18.04
How To Set Up Password Authentication with Apache on Ubuntu 16.04
(same setup as malicious web server access bad actor)
Logs:
[Wed Dec 09 18:57:55.208340 2020] [auth_basic:error] [pid 793:tid 140073706153728] [client 73.239.143.254:53720] AH01618: user badmin not found: /
[Wed Dec 09 18:58:00.023046 2020] [auth_basic:error] [pid 793:tid 140073697761024] [client 73.239.143.254:53720] AH01618: user oops not found: /
[Wed Dec 09 19:00:23.454050 2020] [auth_basic:error] [pid 792:tid 140073722939136] [client 71.212.209.82:56697] AH01618: user bahsdkja not found: /
[Wed Dec 09 19:00:26.323867 2020] [auth_basic:error] [pid 792:tid 140073714546432] [client 71.212.209.82:56697] AH01618: user adf not found: /
[Wed Dec 09 19:00:33.336769 2020] [auth_basic:error] [pid 792:tid 140073680975616] [client 71.212.209.82:56699] AH01618: user afga not found: /
[Wed Dec 09 19:00:45.223817 2020] [auth_basic:error] [pid 793:tid 140073731331840] [client 212.102.47.66:63684] AH01618: user applebottom jeans not found: /

Procedure:
Access the website, and when prompted to enter username and password, enter bad username and a password. Attempt to login.
Logs are at: /var/log/apache2/error.log

Screenshot:
Figure 5. Unauthorized Web Server Logins Logs - Screenshot

4.1.e Port Scan
Logs:
Feb 11 18:11:07 travis sshd[3059]: error: Protocol major versions differ: 2 vs. 1
Feb 11 18:11:07 travis sshd[3060]: error: Protocol major versions differ: 2 vs. 1
Feb 11 18:11:07 travis sshd[3061]: Unable to negotiate with 73.239.143.254 port 55766: no matching host key type found. Their offer: ssh-dss [preauth]
Feb 11 18:11:08 travis sshd[3063]: Connection closed by 73.239.143.254 port 55770 [preauth]
Feb 11 18:11:08 travis sshd[3066]: error: Protocol major versions differ: 2 vs. 1
Feb 11 18:11:08 travis sshd[3067]: error: Protocol major versions differ: 2 vs. 1
Feb 11 18:11:08 travis sshd[3069]: Unable to negotiate with 73.239.143.254 port 55795: no matching host key type found. Their offer: ssh-dss [preauth]
Feb 11 18:11:08 travis sshd[3065]: Connection closed by 73.239.143.254 port 55773 [preauth]
Feb 11 18:11:08 travis sshd[3071]: Unable to negotiate with 73.239.143.254 port 55799: no matching host key type found. Their offer: ecdsa-sha2-nistp384 [preauth]
Feb 11 18:11:09 travis sshd[3075]: Unable to negotiate with 73.239.143.254 port 55804: no matching host key type found. Their offer: ecdsa-sha2-nistp521 [preauth]
Feb 11 18:11:09 travis sshd[3072]: Connection closed by 73.239.143.254 port 55802 [preauth]
Feb 11 18:11:09 travis sshd[3077]: Connection closed by 73.239.143.254 port 55806 [preauth]
Feb 11 18:11:09 travis sshd[3078]: Connection closed by 73.239.143.254 port 55808 [preauth]
Feb 11 18:11:09 travis sshd[3081]: Unable to negotiate with 73.239.143.254 port 55811: no matching host key type found. Their offer: ecdsa-sha2-nistp384 [preauth]
Feb 11 18:11:09 travis sshd[3083]: Unable to negotiate with 73.239.143.254 port 55812: no matching host key type found. Their offer: ecdsa-sha2-nistp521 [preauth]
Feb 11 18:11:10 travis sshd[3085]: Connection closed by 73.239.143.254 port 55813 [preauth]
Dec  6 16:43:39 travis sshd[4416]: error: kex_exchange_identification: Connection closed by remote host

Procedure:
Locate public IP of the network you want to port scan, then enter that IP into a port scanning tool. Logs are at: /var/log/auth.log.




Screenshot: 

Figure 6. Port Scan Logs - Screenshot

4.1.f DDoS
Setup: How To Install the Apache Web Server on Ubuntu 18.04 (same as malicious web server access and unauthorized web server logins)

Logs:
97.126.113.195 - - [06/Jan/2021:18:34:17 -0800] "GET / HTTP/1.0" 200 451 "-" "ApacheBench/2.3"

Repeat above log many times each second. Time is the only difference between lines.

Procedure: 
Simulate attack by launching a “dumb”/simple DDoS attack, that is not being mitigated against on a Apache web server. Look in server logs for a 503 “Service Unavailable” error/flood of requests outside of expected and manageable amount of traffic. For a simple attack all the source IPs can be the same and thus addressed more quickly. Otherwise a flood of requests entails many being made in a short period of time. Then if at any point the upper limits for the number of requests being made is being reached then a warning flag can be raised. Requests in question being explored are along the lines of search requests for data stored on server or account creation requests. Alternatively cmd line launched attacks.
In a linux virtual machine command prompt the DDoS attack can be simulated by entering the command: ab -n 1000 -c 100 http://Enter_IP_Here/
The number following -n specifies the number of requests, and the number following -c specifies the number of concurrent requests.
Logs are at: /var/log/apache2/error.log




Screenshot: 
Figure 7. DDoS Logs - Screenshot

Figure 8. Normal Use Web Server Log
4.2 Log Processing
Logs are read from a text file, and if applicable, an array of labels corresponding to the logs, are also read from a text file. The logs and their corresponding label have to be in the same order. 
Log processing has been explored in two directions: using regex in Python to determine which components of the log message give what information and parsing the log message by spaces and punctuation (similar to natural language processing). The method that is used for this project is to parse the log message by spaces and punctuation.





Figure 9. Example regex parameterization

 The parameterized data will then be converted to a numerical form using a Bag of Words model and an algorithm called term frequency-inverse document frequency (tf-idf). This algorithm weighs the importance of a feature based on the number of occurrences of a feature, and the weight is decreased if the feature is common among the dataset as a whole. This causes features that are uncommon in the dataset, but common in a few logs, to be heavily weighted. tf-idf was implemented using the TfidfVectorizer from the sklearn library.


Figure 10. tf-idf example

4.3 Machine Learning
The machine learning pipeline can be seen in Figure 2. Binary classification is classifying if a log represents a bad actor or not. Bad actor classification is classifying what bad actor a log represents, which is why only logs that have already been classified as a bad actor are examined. Bad actor grouping is looking at all logs and grouping them together.  When new groupings are found, they can be examined to see if they represent a new bad actor or if they are safe.
The machine learning algorithm Support Vector Machines (SVM) was trained by the dataset we constructed of cybersecurity risks is used to perform binary and bad actor classification. This was coded in Python. Several different machine learning algorithms were tested by training the algorithm on a set of logs, then testing the algorithms on a separate set of logs. The algorithm that classified logs with the most accuracy was selected for use in this project. The algorithms tested were neural networks, decision trees, naive bayes, and SVM. K-means has been tested for bad actor grouping, but cannot be used for binary or bad actor classification (yet).
4.3.a Neural Networks
	This implementation was done using the Python libraries TensorFlow and Keras. Keras is used to set up the model and define how it will be evaluated, then TensorFlow runs and predicts the model. The highest accuracy achieved using this model was 94% for binary classification and 90% for bad actor classification
4.3.b Decision Trees
This implementation was done using the Python library sklearn. The tree’s max depth and minimum leaf samples can be adjusted to achieve the best results. The program automatically outputs the accuracy of the algorithm. The highest accuracy achieved using this model was 94% for binary classification and 85% for bad actor classification.
4.3.c Naive Bayes Classifier
	This implementation was done using the Python libraries Scikit-Learn (sklearn), numpy, and pandas. The GaussianNB model of this classifier was used due to the probable shape of the data, its acceptance of floating point values, and the binomial classification of the data.The highest accuracy achieved using this model was 90% for binary classification and 50% for bad actor classification.

4.3.d Support Vector Machines (SVM)
This implementation was done using Scikit-Learn (sklearn). An rbf (radial basis function) kernel was used. This allows for modeling non-linear relationships between the data. The highest accuracy achieved using this model was 100% for binary classification and 98% for bad actor classification.
4.3.e K-Means
This implementation was done using the Kmeans sub-library from the Python library sklearn. The program outputs an array of values corresponding to which cluster a given log message from the testing set was matched with.
4.4 Human Interface Design
Working with this program should be straightforward for Boeing operators. Giving log data to the program needs to be a simple process. Results from the scan should be clearly formatted and easy to analyze. The program should prompt for a text file input of logs, and then it should write a text file that copies the original file, but adds a classification to each line of the text file describing whether the line is classified as safe, or what bad actor it represents, and possibly further, the recommended procedure to deal with a given threat. Additionally, an unsupervised algorithm will be run that will group the data into categories. When new categories start appearing, it indicates that there are potentially new bad actors that the other algorithms should be trained to detect. This program would be used in conjunction with existing bad actor detection programs to ensure nothing is missed.

5. Project Bill of Materials (BOM)
This section describes required components for this project. Cost details are also included in this section.

Table 1. Each revision, and information about the revision.
Item
Price
link
VMware Workstation 16 Player Commercial Licenses
~$596 ($149 per license and four members on the team)
https://store-us.vmware.com/vmware-workstation-16-player-5424180700.html?theme=2



6. Ethical Considerations
This section describes the ethical implications of this project. Mitigation strategies for dealing with ethical issues are also found within this section.

The purpose of this project is to create a proof of concept program that identifies bad actors (logs that potentially indicate a threat to an airplane). In logs generated by aircraft and their subsystems. Failure to identify bad actors could result in a threat to an aircraft’s computing systems going unnoticed, potentially resulting in a threat to the aircraft as a whole.
This issue can only be mitigated by thoroughly testing our program and documenting its accuracy. This will ensure Boeing is not misinformed as to the accuracy of our program so that when they create their own program they can have appropriate expectations as to the accuracy of detections.
Our program does not work with any actual aircraft logs, but if Boeing recreates our program with actual aircraft logs, then the level of privacy/security in which to handle the log data, and thus how the log data is accessed, needs to be approached carefully. Boeing handles this by only allowing authorized personnel to access logs specifically for the purpose of detecting bad actors.

7. References
This section details previous work and specifications that this project relies on. These components are necessary to run the project as designed.

Simulating logs in a Linux Ubuntu environment.
Used support vector machines algorithm to create the program to analyze logs.
Machine learning and log parameterization is done in python.
Log parameterization is done using a python regex library “re”
Term frequency-inverse document frequency feature numerization done using regex library “numpy”
Riley, Sean. “Running an SQL Injection Attack.” YouTube, Computerphile, 15 June 2016, www.youtube.com/watch?v=ciNHn38EyRc. 
Ellingwood, Justin, and Kathleen Juell. “How To Install the Apache Web Server on Ubuntu 18.04.” DigitalOcean, 27 Apr. 2018, www.digitalocean.com/community/tutorials/how-to-install-the-apache-web-server-on-ubuntu-18-04. 
Anderson, Melissa. “How To Set Up Password Authentication with Apache on Ubuntu 16.04.” DigitalOcean, 26 July 2016, www.digitalocean.com/community/tutorials/how-to-set-up-password-authentication-with-apache-on-ubuntu-16-04. 
“Tf–Idf.” Wikipedia, Wikimedia Foundation, en.wikipedia.org/wiki/Tf%E2%80%93idf.
Tripathi, Mayank. “How to Process Textual Data Using tf-idf in Python.” FreeCodeCamp.org, FreeCodeCamp.org, 24 July 2019, www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/. (used for tf-idf figure)


8. Errata
This section details what goals are not able to be met within this achievement. Future projects could work on implementing these goals.

One could train the model using multiple logs as one feature to create a more context-aware model. This has not been tested in this project.
