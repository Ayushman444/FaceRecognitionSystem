import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Flatten,Dense,Input,MaxPooling2D
import uuid  
from layers import *
from data_preProcessing import *
from Model.siamese_network import *

##VERIFICATION FUNCTION

class Verification:

    def __init__(self,path):
        model = load_model()
        self.model = model

    
    def verify(self, detection_threshold, verification_threshold):
        #Build results array
        results=[]
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img=preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_img=preprocess(os.path.join('application_data','verification_images',image))
            result= self.model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
            results.append(result)

        detection=np.sum(np.array(results)>detection_threshold)
        verification=detection/len(os.listdir(os.path.join('application_data','verification_images')))
        verified=verification>verification_threshold
        
        return results,verified
    
    def live_verification(self):
        ##OpenCV Real Time Verification
        cap =cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame=cap.read()
            
            #cut down the frame to 250x250
            frame=frame[120:120+250,200:200+250, :]
            cv2.imshow('Verification',frame)
            
            
            #verification trigger
            if cv2.waitKey(10) & 0xFF == ord('v'):
                #Save input image to input image folder
                cv2.imwrite(os.path.join('application_data','input_image','input_image.jpg'),frame)
        #   Run verification
                results,verified=self.verify(self.model,0.9,0.71)
                print(verified)
                    
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()    

        
