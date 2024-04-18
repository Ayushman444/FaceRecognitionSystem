#MODEL ENGINEERING


## Build Embedding Layer
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
mod.summary()

def make_embedding():
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
    
    # I want the input layer to have 100x100 pixels and 3 channels
    return Model(inputs=[inp],outputs=[d1],name='embedding')

embedding=make_embedding()
embedding.summary()

#Siamese L1 Distance class
class L1Dist(Layer):
    #Init method-inheritance
    def __init__(self,**kwargs):
        super().__init__();
        
    #Magic happens here- similarity calculation
    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding-validation_embedding)

input_image=Input(name="input_img",shape=(105,105,3))
validation_image=Input(name="validation_img",shape=(105,105,3))

inp_embedding=embedding(input_image)
val_embedding=embedding(validation_image)

siamese_layer=L1Dist()
siamese_layer(inp_embedding,val_embedding)

def make_siamese_model():
    #Handle Inputs
    
    #Anchor image input in the network
    input_image=Input(name="input_img",shape=(105,105,3))
    
    #Validation image in the network
    validation_image=Input(name="validation_img",shape=(105,105,3))
    
    #Combine the Siamese Distance components
    siamese_layer=L1Dist()
    siamese_layer._name="distance"
    distances=siamese_layer(embedding(input_image),embedding(validation_image))
    
    #Classification Layer
    classifier=Dense(1,activation='sigmoid')(distances)
    
    return Model(inputs=[input_image,validation_image],outputs=classifier,name='SiameseNetwork')