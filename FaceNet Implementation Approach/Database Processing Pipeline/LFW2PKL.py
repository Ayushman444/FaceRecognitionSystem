#IMPORTS
import pandas as pd
import cv2
import numpy as np
import os
import pickle
from DataBase.Train_Datasets.LFW.lfw import *
import zipfile


#This fetches and downloads the lfw dataset as a zip file
get_lfw()

#This unzips the file
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
  zip_ref.extractall('') #Place the path to original destination here


class ProcessLFW:
   
  df = pd.read_csv('lfw_allnames.csv')

  # Initialing two empty DataFrame with column names
  cols = ['image', 'person']
  dataSet = pd.DataFrame(columns=cols)

  def processImg(self,path_of_folder,input_size=(160,160)):
      
      face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      subFolds = os.listdir(path_of_folder)

      imageSet = []
      classSet = []

      for sub in subFolds:
          paths = os.listdir(sub)
          for path_of_img in paths:
              varis = path_of_img
              path_of_img = str(path_of_folder + "/" + path_of_img)
              image = cv2.imread(path_of_img)
              gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              try:
                (x, y, w, h) = face_detector.detectMultiScale(gray, 1.3, 5)[0]
              except:
                continue
              image = image[y:y+h, x:x+w, :]
              img = cv2.resize(image, input_size)
              img = np.asarray(img)
              img = img.astype('float32') / 255.0
              l = varis.split('.')
              l[0] = l[0].replace('_', ' ')
              l[0] = l[0][:len(l[0]) - 5]
              imageSet.append(img)
              classSet.append(l[0])
      
      self.dataSet.image = imageSet       
      self.dataSet.person = classSet

      with open('lfw_database_processed.pkl', 'wb') as f:
        pickle.dump(self.dataSet, f)
      
      print('Successfully prepared database!!')
    








