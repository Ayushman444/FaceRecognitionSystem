import numpy as np
import pandas as pd
import keras
import pickle
import random

class load_and_refine_data:

  def load_data(self):

    with open('data_test.pkl', 'rb') as f:
      test_df = pickle.load(f)

    with open('data_train.pkl', 'rb') as f:
      train_df = pickle.load(f)

    x_train = train_df['image']
    y_train = train_df['person']

    x_test = test_df['image']
    y_test = test_df['person']

    # Loop through each image in the Series
    for index, image in x_train.items():  # Access elements using items()
      x_train.loc[index] = self.remove_extra_dim(image)  # Modify Series in-place

    for index, image in x_test.items():  # Access elements using items()
      x_test.loc[index] = self.remove_extra_dim(image)  # Modify Series in-place

    classes = set()
    for i in y_train:
      classes.add(i)

    classes = list(classes)

    return x_train, y_train, x_test, y_test, classes

  def remove_extra_dim(self, image):
    try:
        return np.squeeze(image, axis=0)  # Efficient removal if size 1
    except ValueError:
        # Extra dimension not found, return the original data
        print('failed')
        return image


  def triplet_generator(self, batch_size=64):
    x_train, y_train, x_test, y_test, classes = self.load_data()
    while True:
      a = []
      p = []
      n = []
      for _ in range(batch_size):
        found = True
        while(found):
          pos_neg = random.sample(classes, 2)
          try:
            positive_samples = random.sample(list (x_train [y_train==pos_neg [0]]), 2)
            negative_sample= random.choice(list (x_train [y_train == pos_neg [1]]))
            found = False
          except:
            continue
        a.append(positive_samples [0])
        p.append(positive_samples [1])
        n.append(negative_sample)

      yield ([np.array(a), np.array(p), np.array(n)], np.zeros((batch_size, 1)).astype("float32"))

