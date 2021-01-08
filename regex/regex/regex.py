import re

logs = open('S:/QuickAccess/errorLogs.txt', 'r') #@@@ INSERT LOCATION OF LOG FILE HERE

for line in logs:
    if re.search("^\[",line): #log starts with a square bracket
       # print("This is a error.log log file") #these logs aren't currently used for any bad actor scenario
        line = re.split("\[", line, 1)[1]
        date = re.split("\]", line, 1)[0]
        print("date = " + date)
        line = re.split("\[", line, 1) [1]
        logType = re.split("\]", line, 1) [0]
        print("log type = " + logType)
        line = re.split("\[", line, 1) [1]
        IDs = re.split("\]", line, 1) [0]
        PID = re.split("\s", IDs, 1)[1]
        PID = re.split("\:", PID, 1)[0]
        print("PID = " + PID)
        TID = re.split("\s", IDs, 2)[2]
        print("TID = " + TID)
        if re.search("\[(?=client)", line): #ignore if brackets arent for client
            line = re.split("\[", line, 1)[1]
            line = re.split("\s", line, 1)[1]
            client = re.split("\]", line, 1)[0]
            print("client = " + client)
        line = re.split("\s(?=A)", line, 1)[1] #find space followed by "A" 
        errorCode = re.split("\:", line, 1)[0]
        print("error code = " + errorCode)
        msg = re.split("\:", line, 1)[1]
        print("message =" + msg )
        
        


    elif re.search("^\d",line): #log starts with a digit
        #print("This is a access.log log file") #used for malicious web server access bad actor
        #log format:
        #ip - user (- if not relevant) [date & time] "GET \X"-"(if not needed)
            #else "GET \X" \d \d "http://ip/" "Browser information"
        ip = re.split("\s", line, 1)[0]
        print("ip = " + ip)
        line = re.split("\s", line, 1)[1]
        line = re.split("(?<=-)\s", line, 1)[1]
        user = re.split("\s", line, 1)[0]
        print("user = " + user)
        line = re.split("\s\[", line, 1)[1]
        date = re.split("]", line, 1)[0]
        print("date = " + date)
        msg = re.split("] ", line, 1)[1]
        print("message = " + msg)
        
    elif re.search("^[A-Za-z]",line): #log starts with a alphabetical character
        #print("This is a auth.log log file")
        #log format:
        #month  day hour:minute:second user useraccount logtype: main information
        date = re.split("(?<=.{15})\s", line, 1)[0]
        print("date = " + date)
        line = re.split("(?<=.{15})\s", line, 1)[1]
        user = re.split("\s", line)[0]
        print("user = " + user)
        msg = re.split("\s", line, 1)[1]
        print("message = " + msg)

logs.close()

#print("End of program")

#access.log ip, date time, use quotes to determine next two sections
#error.log date time, basic log type, process & thread id (possibly client also), AH0 something, main log information
#other logs ???

#auth.log: date time, user account, log type, (separated by :) main information
#error.log starts with '[', access.log starts with a digit, auth.log starts with a letter (first letter of month)

#use \d to detect any digit
#use \D to detect any non-digit
#[A-Za-z]
