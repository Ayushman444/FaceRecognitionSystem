'''
THIS SEGMENT IS TO DOWNLOAD THE DATASET FROM THE DRIVE FOLDER TO THE COLAB LOCAL FOLDER AND UNZIP IT
'''
# !rm -rf /content/.cache && rm -rf /tmp/*
# from google.colab import drive

# drive.mount('/content/drive')

# import zipfile

# with zipfile.ZipFile('/content/drive/MyDrive/archive.zip', 'r') as zip_ref:
#   zip_ref.extractall('/content/')

# drive.flush_and_unmount()


#IMPORTS
import pandas as pd
import cv2
import numpy as np
import os
from math import floor
import pickle


df = pd.read_csv('lfw_allnames.csv')
to_explore = []

for i,j in zip(df.images,df.name):
  if(i>2):
    to_explore.append(j)



# Initialing two empty DataFrame with column names
cols = ['image', 'person']
trainSet = pd.DataFrame(columns=cols)
testSet = pd.DataFrame(columns=cols)

def processImg(path_of_folder):
    face_classifier = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
    paths = os.listdir(path_of_folder)
    imageTrainArr = []
    imageTestArr = []
    classesTrain = []
    classesTest = []

    count = 0

    factor = 0.8
    fetchLen = len(paths)

    trainLen = floor(factor*fetchLen)
    testLen = fetchLen-trainLen
    personDrop = 0
    iterab = 0
    # print(fetchLen,trainLen,testLen,sep=',')
    for path_of_img in paths:
        varis = path_of_img
        path_of_img = str(path_of_folder + "/" + path_of_img)
        image = cv2.imread(path_of_img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
          (x, y, w, h) = face_classifier.detectMultiScale(gray, 1.3, 5)[0]
        except:
          count+=1
          fetchLen-=1
          if(fetchLen<3):
            personDrop+=1
            break
          else:
                trainLen = floor(factor*fetchLen)
                testLen = fetchLen-trainLen
          continue

        image = image[y:y+h, x:x+w, :]
        img = cv2.resize(image, (224, 224))
        img = np.asarray(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        l = varis.split('.')
        l[0] = l[0].replace('_', ' ')
        l[0] = l[0][:len(l[0]) - 5]
        iterab+=1

        if(iterab<=trainLen):
            imageTrainArr.append(img)
            classesTrain.append(l[0])
        else:
            imageTestArr.append(img)
            classesTest.append(l[0])

    return (imageTrainArr,classesTrain,imageTestArr,classesTest,count,personDrop)

inter = '/content/lfw-deepfunneled/lfw-deepfunneled/'

all_images_train = []
all_images_test = []
all_classes_train = []
all_classes_test = []
netRem = 0
netDrop = 0

for i in to_explore:
  try:
    imageTrainArr,classesTrain,imageTestArr,classesTest,count,perDrop = processImg(inter+i)
    all_images_train.extend(imageTrainArr)
    all_classes_train.extend(classesTrain)
    all_images_test.extend(imageTestArr)
    all_classes_test.extend(classesTest)
    netRem = netRem + count
    netDrop = perDrop + netDrop

  except:
    continue
print('removed image: ',netRem)

print(netDrop)

print(len(all_images_train))

print(len(all_classes_train))

print(len(all_images_test))

print(len(all_classes_test))

s = set()
for i in all_classes_train:
  s.add(i)
len(s)

s2 = set()
for i in all_classes_test:
  s2.add(i)
len(s2)

trainSet.image[0].shape

with open('data_train.pkl', 'wb') as f:
    pickle.dump(trainSet, f)

with open('data_test.pkl', 'wb') as f:
    pickle.dump(testSet, f)


'''
THIS SEGMENT IS TO STORE BACK THE PKL FILES TO THE DRIVE
'''
# from google.colab import drive

# drive.mount('/content/drive')

# !cp data_train.pkl /content/drive/MyDrive/Dataset/data_train.pkl
# !cp data_test.pkl /content/drive/MyDrive/Dataset/data_test.pkl

# drive.flush_and_unmount()