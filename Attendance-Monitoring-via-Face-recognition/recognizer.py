import cv2
import numpy as np
from datetime import datetime 
import csv

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model/trained_model.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
#choosing font
font = cv2.FONT_HERSHEY_SIMPLEX

cam = cv2.VideoCapture(0)


now=datetime.now()
file=open(str(now.year)+"_"+str(now.month)+"_"+str(now.day)+".csv", 'w+',newline='')
csvwriter = csv.writer(file)
csvwriter.writerow(['name','rollno','status'])

f=open("student_record.csv")
readers=csv.reader(f)

ids=[]
name={}

for row in readers:
      print(row[0])
      ids.append(int(row[0]))
      name[int(row[0])]=row[1]
while True:
    ret, im =cam.read()

    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    
    for(x,y,w,h) in faces:

        Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        #print(conf)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
        if(conf<=48):
            cv2.putText(im, str(Id), (x,y-40),font, 2, (255,255,255), 3)
            if Id in ids:
                csvwriter.writerow([name[Id],Id,'present'])
                ids.remove(Id)

    cv2.imshow('im',im)
    if cv2.waitKey(1)==13:
        break
for i in ids:
    csvwriter.writerow([name[i],i,'absent'])
f.close()
file.close()
cam.release()
cv2.destroyAllWindows()