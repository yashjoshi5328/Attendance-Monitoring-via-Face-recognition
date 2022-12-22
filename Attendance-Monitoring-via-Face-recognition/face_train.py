import cv2
import numpy as np
import os
import csv 

f=open("student_record.csv",'a',newline='')
csvwriter = csv.writer(f)


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_cap():

    cap = cv2.VideoCapture(0)

    ID = input("Enter your ID: ")

    Name = input("Enter your name: ")
    
    csvwriter.writerow([ID,Name])
    
    f.close()

    sampleNum = 0

    while (True):

        ret,img = cap.read()
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
            # break if the sample number is morethan 1000
        elif sampleNum > 1000:
            break

    print("\nFACE DATA SAVED FOR {0} , ID : {1}".format(Name,ID))   
    cap.release()
    cv2.destroyAllWindows() 
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

        
def train_model():

    recognizer= cv2.face.LBPHFaceRecognizer_create()
    
    faces,ID=get_img_and_id("face_data")

    recognizer.train(faces, np.array(ID))
    
    recognizer.save("model/trained_model.yml")

    print("MODEL TRAINED")


face_cap()     
train_model()

