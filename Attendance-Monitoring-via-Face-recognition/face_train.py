import cv2
import numpy as np
import os
import csv 
from tkinter import *

f=open("student_record.csv",'a',newline='')
csvwriter = csv.writer(f)


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def get_img_and_id(path):
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    #storing all images path present in the path directory in imamgepaths list
    imagePaths=[]
    #list dir gives list of all image present in path directory
    for f in os.listdir(path):
        #now single image name will join with its path and append to imagepaths list
        imagePaths.append(os.path.join(path,f))
    for imagePath in imagePaths:
        #reading image
        image=cv2.imread(imagePath)
        #converting image to gray
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
         # Now we are converting the gray image into numpy array
        imageNp=np.array(gray,'uint8')
        #extracting id name from path                                                                      
        '''
        os.path.split(name)   
        =======('Face-Recognition/dataSet', 'face-1.1.jpg')
        os.path.split(name)[-1]                                                                      
        ======='face-1.1.jpg'
        os.path.split(name)[-1].split('.')                                                           
        =======['face-1', '1', 'jpg']
        os.path.split(name)[-1].split('.')[1]                                                        
        ======='1'
        int(os.path.split(name)[-1].split('.')[1])                                                   
        =======1
        '''
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        #name=os.path.split(imagePath)[-1].split('.')[0]
        faces = face_detector.detectMultiScale(imageNp)
        
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(ID)
           
    return faceSamples, Ids


#face_cap()     
#train_model()
window=Tk() #initiating a instance of a window 

icon=PhotoImage(file='graphic.png')
window.iconphoto(True,icon)

window.geometry('840x420')

window.title('FACE TRAIN SYSTEM')
window.config(background='#D2B4DE')

label1=Label(window,text='FACE TRAIN SYSTEM',font=('Arial',30,'bold','underline'),fg='black',bg='#D2B4DE')
label1.pack()

label2=Label(window,text='Name',font=('Arial',20,'bold'),fg='black',bg='#D2B4DE')
label2.place(x=0,y=100)
entry1=Entry(window,font=('Arial',20,'bold'),fg='black')
entry1.place(x=100,y=100)

label3=Label(window,text='ID',font=('Arial',20,'bold'),fg='black',bg='#D2B4DE')
label3.place(x=0,y=140)
entry2=Entry(window,font=('Arial',20,'bold'),fg='black')
entry2.place(x=100,y=140)

def face_cap():

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3);

    ID = entry2.get()

    Name = entry1.get()
    
    csvwriter.writerow([ID,Name])

    sampleNum = 0

    while (True):

        ret,img = cap.read()
        if(ret):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # incrementing sample number
                sampleNum = sampleNum + 1

                # saving the captured face in the face_data folder with imwrite("name with path",image)
                cv2.imwrite("face_data/ " + Name + "." + ID + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('Frame',img)

            # wait for 100 miliseconds
        if cv2.waitKey(1)== 13 :
            break
            # break if the sample number is more than 1000
        elif sampleNum > 150:
            break

    print("\nFACE DATA SAVED FOR {0} , ID : {1}".format(Name,ID))   
    cap.release()
    cv2.destroyAllWindows() 

button1=Button(window,text='SUBMIT',command=face_cap,font=('Comic Sans',15),activebackground='RED')
button1.place(x=370,y=200)

def train_model():

    recognizer= cv2.face.LBPHFaceRecognizer_create()
    
    faces,ID=get_img_and_id("face_data")

    recognizer.train(faces, np.array(ID))
    
    recognizer.save("model/trained_model.yml")

    print("MODEL TRAINED")

button2=Button(window,text='UPDATE MODEL',command=train_model,font=('Comic Sans',15),activebackground='RED')
button2.place(x=330,y=250)

window.mainloop() #display window 

f.close()
