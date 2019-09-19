import numpy as np
import cv2
import pickle
h={}
h1={}
with open('labelsnow.pickle','rb') as f:
    h=pickle.load(f)
print(h)
h1={v:k for k,v in h.items()}
print(h1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainnernow.yml')
cap = cv2.VideoCapture(1)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        id_,conf=recognizer.predict(roi_gray)
        print(id_)
                
        if(conf>=20 and conf<=90):
            '''
            for key,value in h.items():
                if(value==id_):
                    ret=key
                    break
            '''
            print('hello')
            #print(h1[id_])
            #cv2.putText(img,h1[id_],(x,y),cv2.FONT_HERSHEY_DUPLEX,color=(0,255,0),4,cv2.LINE_AA)
            cv2.putText(img=img,text=h1[id_],org=(x-1,y-1),fontFace=cv2.FONT_ITALIC,fontScale=2,color=(0,255,0))
        
        
    cv2.imshow('frame2',img)
    if(cv2.waitKey(50) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
    
