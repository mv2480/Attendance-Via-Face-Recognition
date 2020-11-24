import numpy as np
import cv2
import face_recognition as fc
import os
from datetime import datetime
path = 'Attendace Images'
images = []
names = []
mylist = os.listdir(path)
print(mylist)

for cls in mylist:
    curImage = cv2.imread(f'{path}/{cls}')
    images.append(curImage)
    names.append(os.path.splitext(cls)[0])
print(names)

def MarkAttendance(name):
    with open('Resources/Attendance.csv', 'r+') as f:
        DataList = f.readlines()
        Atten = []
        for line in DataList:
            entry = line.split(',')
            Atten.append(entry[0])
        if name not in Atten:
            now = datetime.now()
            Dtime = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{Dtime}')

def Encodings(images) :
    AllEncodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        AllEncodings.append(fc.face_encodings(img)[0])
    return AllEncodings
allEncodings = Encodings(images)

cap = cv2.VideoCapture(0)



while True:
    success , img = cap.read()
    #img = cv2.resize(img,(0,0),None,0.25,0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    CurFrameLocation = fc.face_locations(img)
    CurFrameEncodings = fc.face_encodings(img,CurFrameLocation)
    result = []
    faceDis = []
    for Loc,encodes in zip(CurFrameLocation,CurFrameEncodings):
        result = fc.compare_faces(allEncodings,encodes)
        faceDis = fc.face_distance(allEncodings,encodes)
        print(faceDis)
        index = np.argmin(faceDis)
        if faceDis[index] :
            print(names[index])
            y1 , x2, y2, x1 = Loc
            #y1, x2, y2, x1 = y1*4 , x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),1)
            cv2.putText(img,names[index],(Loc[3],Loc[0]),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            MarkAttendance(names[index])
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow("Webcam",img)
    cv2.waitKey(1)



