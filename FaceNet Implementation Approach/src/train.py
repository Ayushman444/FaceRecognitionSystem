import numpy as np
import pandas as pd
import keras
import pickle
import random
from Models.FaceNet_second_model_proposition import *
from data_generator import *

''' THIS CODE SEGMENT IS TO DOWNLOAD THE LFW DATA ALREADY PROCESSED BY LFW2PKL.py'''
# from google.colab import drive
# class Drive:
#     def __init__(self):
#       drive.mount('/content/drive', force_remount = True)
#       !cp /content/drive/MyDrive/Dataset/data_train.pkl /content
#       !cp /content/drive/MyDrive/Dataset/data_test.pkl /content
#       drive.flush_and_unmount()
# driver = Drive()

# class load_and_refine_data:

#   def load_data(self):

#     with open('data_test.pkl', 'rb') as f:
#       test_df = pickle.load(f)

#     with open('data_train.pkl', 'rb') as f:
#       train_df = pickle.load(f)

#     x_train = train_df['image']
#     y_train = train_df['person']

#     x_test = test_df['image']
#     y_test = test_df['person']

#     # Loop through each image in the Series
#     for index, image in x_train.items():  # Access elements using items()
#       x_train.loc[index] = self.remove_extra_dim(image)  # Modify Series in-place

#     for index, image in x_test.items():  # Access elements using items()
#       x_test.loc[index] = self.remove_extra_dim(image)  # Modify Series in-place

#     classes = set()
#     for i in y_train:
#       classes.add(i)

#     classes = list(classes)

#     return x_train, y_train, x_test, y_test, classes

#   def remove_extra_dim(self, image):
#     try:
#         return np.squeeze(image, axis=0)  # Efficient removal if size 1
#     except ValueError:
#         # Extra dimension not found, return the original data
#         print('failed')
#         return image


#   def triplet_generator(self, batch_size=64):
#     x_train, y_train, x_test, y_test, classes = self.load_data()
#     while True:
#       a = []
#       p = []
#       n = []
#       for _ in range(batch_size):
#         found = True
#         while(found):
#           pos_neg = random.sample(classes, 2)
#           try:
#             positive_samples = random.sample(list (x_train [y_train==pos_neg [0]]), 2)
#             negative_sample= random.choice(list (x_train [y_train == pos_neg [1]]))
#             found = False
#           except:
#             continue
#         a.append(positive_samples [0])
#         p.append(positive_samples [1])
#         n.append(negative_sample)

#       yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))

face_net = FaceNet_Model()

my_optimizer = keras.optimizers.Adam(0.0005)
model_to_use = face_net.facenet()

model_to_use.compile(loss=face_net.triplet_loss, optimizer=my_optimizer)
get_data = load_and_refine_data()

model_to_use.fit_generator(get_data.triplet_generator(),steps_per_epoch=87,epochs=10)

'''THIS CODE SEGMENT IS USED TO STORE THE TRAINED MODEL INTO DRIVE'''
# class Save_To_Drive:

#   model_path = "/content"

#   def __init__(self, model):

#     tf.keras.models.save_model(model,self.model_path)
#     model.save('facenetep10.h5',save_format='h5')

#     drive.mount('/content/drive')

#     !cp /content/facenetep10.h5 /content/drive/MyDrive

#     !cp /content/fingerprint.pb /content/FaceNet
#     !cp /content/saved_model.pb /content/FaceNet
#     !cp /content/keras_metadata.pb /content/FaceNet
#     !cp /content/variables/variables.data-00000-of-00001 /content/FaceNet/variables
#     !cp /content/variables/variables.index /content/FaceNet/variables

#     import os
#     from zipfile import ZipFile
#     with ZipFile("FaceNetep10.zip",'w') as zip_file:
#       for root,_,files in os.walk('/content/FaceNet'):
#         for filer in files:
#           filepath = os.path.join(root,filer)
#           zip_file.write(filepath,os.path.relpath(filepath,'/content/FaceNet'))

#     print('success!')

#     !cp /content/FaceNetep10.zip /content/drive/MyDrive
#     drive.flush_and_unmount()
# Save_To_Drive(model_to_use)

