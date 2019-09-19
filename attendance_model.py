import os
import numpy as np
import pickle
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
base_dir=os.path.dirname(os.path.abspath(__file__))
#print(base_dir)
img_dir=os.path.join(base_dir,'imagesnow')
recognizer=cv2.face.LBPHFaceRecognizer_create()
#print(img_dir)
current_id=0
label_ids={}
xtrain=[]
ylabel=[]
for root,dirs,files in os.walk(img_dir):
    #print('root',root)
    #print('dir',dirs)
    #print('files',files)
    for file in files:
        if(file.endswith('png') or file.endswith('jpg')):
            path=os.path.join(root,file)
            label=os.path.basename(root)
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                print(current_id)
                current_id=current_id+1
            id_=label_ids[label]
            
            
            pil_image=Image.open(path).convert('L')
            final_image=pil_image.resize((640,480),Image.ANTIALIAS)
            image_array=np.array(pil_image,'uint8')
            #print(image_array)
            
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)
            for (x,y,w,h) in faces:
                roi= image_array[y:y+h, x:x+w]  
                xtrain.append(roi)
                ylabel.append(id_)
print(label_ids)                
with open('labelsnow.pickle','wb') as f:
    pickle.dump(label_ids,f)
    
recognizer.train(xtrain,np.array(ylabel))
recognizer.save('trainnernow.yml')    
print('hello')
#print(xtrain)
#print(ylabel)