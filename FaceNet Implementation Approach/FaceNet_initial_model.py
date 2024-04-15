import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

input_shape = (224,224,3);

#This FaceNet Implementation using Inception_V3
input_layer = layers.Input(shape=input_shape)
inception = InceptionV3(weights='imagenet', include_top=False)
feature_extractor = inception(input_layer)
for layer in inception.layers:
    layer.trainable = False
gap_layer = layers.GlobalAveragePooling2D()(feature_extractor)
fc_layer = layers.Dense(128, activation="relu")(gap_layer)
l2_norm_layer = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(fc_layer)
model = Model(inputs=input_layer, outputs=l2_norm_layer)

model.summary()