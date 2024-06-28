import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import cv2
from Models.FaceNet_third_model_proposition_using_preTrained_fineTuning import *
from sklearn.preprocessing import Normalizer
import os



l2_normalizer = Normalizer('l2')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = Embed_model()
paths = os.listdir("A:\\Facial Recognition System - Netra\\Face Recognition System - Netra\\FaceNet_Implementation_Approach\\Image_Dataset")

data = []
for i in paths:
    if(i[0]!='t'):
        image = cv2.imread("A:\\Facial Recognition System - Netra\\Face Recognition System - Netra\\FaceNet_Implementation_Approach\\Image_Dataset\\"+i)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            gray = image
        try:
            (x, y, w, h) = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)[0]
            face_img = image[y:y+h, x:x+w]
        except:
            continue
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Get the face embedding
        face_embedding = model.predict(face_img)[0]
        face_embedding = l2_normalizer.transform(np.expand_dims(face_embedding, axis=0))[0]
        data.append((face_embedding,i[0:len(i)-4]))

with open('faceEmbeddings.pkl','wb') as f:
    pickle.dump(data,f)
