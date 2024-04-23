import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Conv2D,Flatten,Dense,Input,MaxPooling2D
from layers import *
from src.data_preProcessing import *

##Build Train and Test Partition
def preprocess_twin(input_img,validation_img,label):
    return (preprocess(input_img),preprocess(validation_img),label)

res=preprocess_twin(*examples) # *examples unpacks what is there in the examples variable

plt.imshow(res[1])

#Build dataloader pipeline
data=data.map(preprocess_twin)
data=data.cache() #cachine for speed retrieval of the dataset, especially during large dataset
data=data.shuffle(buffer_size=1024)

#training partition
train_data=data.take(round(len(data)*.7))
train_data=train_data.batch(16)
train_data=train_data.prefetch(8)
#The 8 specifies the number of batches to prefetch. So, while the
#model is processing one batch of data (which takes some time, especially on GPUs), 
#the data pipeline is already preparing the next 8 batches in the background. 

train_samples=train_data.as_numpy_iterator()

train_sample=train_samples.next()

#testing partition
test_data=data.skip(round(len(data)*.7))
test_data=test_data.take(round(len(data)*.3))
test_data=test_data.batch(16) 

#MODEL ENGINEERING
## Build Embedding Layer

class Siamese_net:
    
    def architecture(self,summary=False):

        inp=Input(shape=(105,105,3),name='input_image')
        inp #Maybe you can change to 100,100,3 later for the model testing

        c1=Conv2D(64,(10,10),activation='relu')(inp)
        print(c1)
        m1=MaxPooling2D(64,(2,2),padding='same')(c1)
        print(m1)

        #Second block
        c2=Conv2D(128,(7,7),activation='relu')(m1)
        print(c2)
        m2=MaxPooling2D(64,(2,2),padding='same')(c2)
        print(m2)

        c3=Conv2D(128,(4,4),activation='relu')(m2)
        print(c3)
        m3=MaxPooling2D(64,(2,2),padding='same')(c3)
        print(m3)

        c4=Conv2D(256,(4,4),activation='relu')(m3)
        print(c4)
        f1=Flatten()(c4)
        print(f1)
        d1=Dense(4096,activation='sigmoid')(f1)
        print(d1)
        mod=Model(inputs=[inp],outputs=[d1],name='embedder')

        if(summary):
            print(mod.summary())

    def make_embedding(self,summary=False):
        inp=Input(shape=(105,105,3),name='input_image')
        
        #First block
        c1=Conv2D(64,(10,10),activation='relu')(inp)
        m1=MaxPooling2D(64,(2,2),padding='same')(c1)
        
        
        #Second block
        c2=Conv2D(128,(7,7),activation='relu')(m1)
        m2=MaxPooling2D(64,(2,2),padding='same')(c2)
        
        #Third block
        c3=Conv2D(128,(4,4),activation='relu')(m2)
        m3=MaxPooling2D(64,(2,2),padding='same')(c3)
        
        #Final Embedding Block
        c4=Conv2D(256,(4,4),activation='relu')(m3)
        f1=Flatten()(c4)
        d1=Dense(4096,activation='sigmoid')(f1)
        
        embedding_model = Model(inputs=[inp],outputs=[d1],name='embedding')

        if(summary):
            print(embedding_model.summary())

        # I want the input layer to have 100x100 pixels and 3 channels
        return embedding_model
    
    def siamese_model(self,summary=False):
        #Anchor image input in the network
        input_image=Input(name="input_img",shape=(105,105,3))
        
        #Validation image in the network
        validation_image=Input(name="validation_img",shape=(105,105,3))
        
        #Combine the Siamese Distance components
        siamese_layer=L1Dist()
        siamese_layer._name="distance"

        embedding = self.make_embedding()
        distances=siamese_layer(embedding(input_image),embedding(validation_image))
        
        #Classification Layer
        classifier=Dense(1,activation='sigmoid')(distances)
        siamese_model = Model(inputs=[input_image,validation_image],outputs=classifier,name='SiameseNetwork')
        
        if(summary):
            print(siamese_model.summary())
            
        return siamese_model
    
class SiameseModel:

    binary_cross_entropy=tf.losses.BinaryCrossentropy()
    op=tf.keras.optimizers.Adam(1e-4)#0.0001
    checkpoints_dir='./training_checkpoints'
    checkpoints_prefix=os.path.join(checkpoints_dir,'ckpt')

    def __new__(self):
        siamese_model = Siamese_net.siamese_model()
        self.model = siamese_model
        return siamese_model

    @tf.function
    def train_step(self,batch):

        '''To generate batches on the fly'''
        # tester=train_data.as_numpy_iterator()
        # batch_1=tester.next()

        with tf.GradientTape() as tape:
        #Get anchor and positive/negative image
            X=batch[:2]
            #Get Label
            Y=batch[2]
            
            #Forward pass
            yhat=self.model(X,training=True)
            #Calculate the loss
            loss=self.binary_cross_entropy(Y,yhat)
        print(loss)    
            
        #Calculate gradient
        grad=tape.gradient(loss,self.model.trainable_variables)
        
        #Calculate the updated weights and apply to siamese model
        self.op.apply_gradients(zip(grad,self.model.trainable_variables))
            
        #return loss    
        return loss
    
    #Build a Training Loop
    def train(self,data,EPOCHS):

        checkpoint=tf.train.Checkpoint(op=self.op,siamese_network=self.model) # allows to save and restore the state of model during training and inference
        # Loop through the EPOCHS
        for epoch in range(1,EPOCHS+1):
            print('\n Epoch {}/{}'.format(epoch,EPOCHS))
            progbar=tf.keras.utils.Progbar(len(data))
        
            # Loop through each batch
            for idx, batch in enumerate(data):
                self.train_step(batch)
                progbar.update(idx+1)
                
            #Save checkpoints 
            if epoch%10==0:
                checkpoint.save(file_prefix=self.checkpoint_prefix)
    
    def save(self):
        self.model.save('siamese_model.h5')
    
def load_model(self,path='siamesemodel.h5'):
    model=tf.keras.models.load_model(path,custom_objects={'L1Dist':L1Dist,'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
    return model




#EVALUATE MODEL
'''THIS CODE SEGMENT DEALS WITH TESTING AND EVALUATION AND ALSO WE DEAL WITH BATCH CREATION'''
# #Import metric calculation
# from tensorflow.keras.metrics  import Precision, Recall

# #Get a batch of test data
# test_input, test_val, y_true=test_data.as_numpy_iterator().next()
# test_var=test_data.as_numpy_iterator().next()


