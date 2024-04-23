#Caution Do Not Run the code.

'''IMPORTS'''
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Flatten,Dense,Input,MaxPooling2D
import uuid  
from database.train_data.trainData import *



'''
What's about the positive, negative and anchor, to be explained!! All this has already been processed and developed
'''



''' This section is to prepare directories for anchor, positive and negative however the same can be downloade now by:

get_trainData()

once you run the above given function please unzip the folder and follow along
'''

#defining the paths or selecting them
POS_PATH=os.path.join('data','positive')
NEG_PATH=os.path.join('data','negative')
ANC_PATH=os.path.join('data','anchor')

#Make Directories
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

#Mov lfw images to the following repository  data/negative
for directory in os.listdir('lfw'): #path to be added 
    for file in os.listdir(os.path.join('lfw',directory)):
        EX_PATH=os.path.join('lfw',directory,file);
        NEW_PATH=os.path.join(NEG_PATH,file)
        os.replace(EX_PATH,NEW_PATH)

'''
This code segment was used initially to generate and save the anchor positive and negative
'''
#Establish a connection with the webcam
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    
    #cut down the frame to 250x250
    frame=frame[120:120+250,200:200+250, :]
    
#Show image back to screen
    cv2.imshow('Image being captured',frame)
    
#Collecting the positives
    if cv2.waitKey(1) & 0XFF==ord('p'):
        #Creating a unique file path 
        imgname=os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1()))
        #Write out positives image
        cv2.imwrite(imgname,frame)
    
#Collecting the anchors
    if cv2.waitKey(1) & 0XFF==ord('a'):
        #Creating a unique file path 
        imgname=os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        #Write out anchor image
        cv2.imwrite(imgname,frame)
    
    
#Breaking gracefully
    if cv2.waitKey(1) & 0XFF==ord('q'):
          break                 

#Release the webcam            
cap.release()
#Close Image Show frame
cv2.destroyAllWindows()

'''LOAD AND PREPROCESS IMAGES'''

#Get the Image Directories
anchor=tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(200) # check what the performance suits like to the model 
positive=tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(200)#takes the path of the dataset in POS_PATH directory and total count = 300
negative=tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(200)
dir_test=anchor.as_numpy_iterator()
print(dir_test.next())


'''This code is for pre-processing of data'''
# #Preprocessing Scale and Resize
def preprocess(filepath):
    #Reads in the image from file path
    byte_img=tf.io.read_file(filepath)
    #laads the image
    img=tf.io.decode_jpeg(byte_img)
    #Preprocessing step- resizing the image in to 100x100x3
    img=tf.image.resize(img,(105,105))
    #Scale an image between 0 and 1 
    img=img/255.0
    return img

#test preprocess
imager=preprocess('data\\anchor\\365f301d-f755-11ee-bdaf-6479f0d51122.jpg')


#Create a labelled Dataset
positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
#This line combines the anchor, positive, and a dataset of ones into a single dataset using tf.data.Dataset.zip(). The ones are likely labels indicating positive examples. It's setting up the data in a format where each sample consists of an anchor image path, a positive image path, and the label (in this case, 1 for positive).
negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor))))) # creates a seperate dataset for 0s
data=positives.concatenate(negatives)
samples=data.as_numpy_iterator()
examples=samples.next()




    

