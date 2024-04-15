import numpy as np
import cv2
from keras_facenet import FaceNet

def preProcess(pathImg):
    img = cv2.imread(pathImg)
    cv2.resize(img,(160,160))
    img = np.asarray(img)
    img = img.astype('float32')/255.0
    img = np.expand_dims(img, axis=0)
    return img

pathe = input('Specify Image Path: ')  #For testing phase
img = preProcess(pathe)
model = FaceNet()
embed = model.embeddings(img)





