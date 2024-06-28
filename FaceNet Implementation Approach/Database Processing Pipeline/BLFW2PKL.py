import os
import zipfile
import pickle
import cv2
from DataBase.Test_Datasets.Balanced_LFW.BLFW import *

class ProcessBLFW:
  # Function to extract zip file
  def extract_zip(self, zip_path, extract_to):
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
          zip_ref.extractall(extract_to)

  # Function to read images and labels from the extracted directory
  def read_blfw_data(self, data_dir):
      data = {}
      for person_name in os.listdir(data_dir):
          person_dir = os.path.join(data_dir, person_name)
          if os.path.isdir(person_dir):
              images = []
              for img_name in os.listdir(person_dir):
                  img_path = os.path.join(person_dir, img_name)
                  if img_path.endswith('.jpg'):
                      img = cv2.imread(img_path)
                      if img is not None:
                          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
                          images.append(img)
              if images:
                  data[person_name] = images
      return data

  # Function to save data as a pickle file
  def save_as_pickle(self, data, output_path):
      with open(output_path, 'wb') as f:
          pickle.dump(data, f)

class BLFWData:
  get_BLFW()
  zip_path = 'datasetBLFW.zip'
  extract_to = os.getcwd()
  output_pickle = 'blfw_data.pkl'

  handler = ProcessBLFW()
  handler.extract_zip(zip_path, extract_to)
  data = handler.read_blfw_data(extract_to)
  handler.save_as_pickle(data, output_pickle)
