#Custom L1 Distance Layer Module
# WHY DO WE NEED THIS:its needed to load the custom model
#import dependencies
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer

#custom L1 Distance from Jupyter
class L1Dist(Layer):
    #Init method-inheritance
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    #Magic happens here- similarity calculation
    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding-validation_embedding)

