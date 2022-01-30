import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
path='C:\\Users\\Hp\\Documents\\sampleclassifier\\imgdwnl'
orb=cv.ORB_create(nfeatures=5000)
#Importing images
images=[]
names=[]
mylist=os.listdir(path)
print("No. of classes are:",len(mylist))
for i in mylist:
    thisimg=cv.imread(f'{path}/{i}',0)
    images.append(thisimg)
    names.append(os.path.splitext(i)[0])
    # print(names)


#To find descriptor of saved images
def Des(images):
    dlist=[]
    for img in images:
        k,d=orb.detectAndCompute(img,None)
        dlist.append(d)
    return dlist


#calling the descriper of the camera frame and matching it with the saved images(for this call the function from video capture below)
def MatchID(ig,list,margin=30):
    k2,d2=orb.detectAndCompute(ig,None)
    bf=cv.BFMatcher()
    matchimg=[]
    indx=-1 
    try:
        for d in list:
            matches=bf.knnMatch(d,d2,k=2)

            goodmatch=[]
            #two variables must be passed through matches since the value k is set to 2 in above knn functino
            for x,y in matches:
                if x.distance < 0.75*y.distance:
                    goodmatch.append([x])
                    #print(goodmatch)
            matchimg.append(len(goodmatch))
        print(matchimg)

    except:
        pass


    if len(matchimg)!=0:
        if max(matchimg)>margin:
            indx=matchimg.index(max(matchimg))

    return indx

        



li=Des(images)
print(len(li))

cap=cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    #capturing video frame by frame
    ret,frame=cap.read()
    ogimg=frame.copy()
    #if frame is read correctly, ret is true
    if not ret:
        print("Frame not recieved correctly....")
        break
    fram=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cmp=MatchID(frame,li)
    if cmp!=-1:
        cv.putText(ogimg,names[cmp],(80,60),cv.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)

    else:
        print("Show some object...")
    cv.imshow('frame',ogimg)
    if cv.waitKey(1)==ord('q'): 
        break
