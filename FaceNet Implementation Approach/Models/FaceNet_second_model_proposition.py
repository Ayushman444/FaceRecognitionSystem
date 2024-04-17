from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.models import load_model


input_shape = (224,224,3);

#This FaceNet Implementation using Inception_V3
input_layer = layers.Input(shape=input_shape)
inception = InceptionV3(weights='imagenet', include_top=False)
feature_extractor = inception(input_layer)
for layer in inception.layers:
    layer.trainable = False
gap_layer = layers.GlobalAveragePooling2D()(feature_extractor)
dropout_layer = layers.Dropout(rate=0.2,seed=82)(gap_layer)
fc_layer = layers.Dense(128, activation='relu')(dropout_layer)
batchNorm = layers.BatchNormalization()(fc_layer)
model = Model(inputs=input_layer, outputs=batchNorm)

#Triplet model for training the weights
triplet_model_a = layers.Input((224, 224, 3))
triplet_model_p = layers.Input((224, 224, 3))
triplet_model_n = layers.Input((224, 224, 3))
triplet_model_out = layers.Concatenate()([model (triplet_model_a), model (triplet_model_p), model(triplet_model_n)])
triplet_model = Model( [triplet_model_a, triplet_model_p, triplet_model_n], triplet_model_out)

#Inference Model to be used for prediction (not to train)
model = load_model('path_to_savedModel_file', custom_objects={'triplet_loss': triplet_loss}) #Triplet loss file to be updated soon
single_image_input = layers.Input(shape=(224, 224, 3), name='single_image_input')
embedding_output = model.layers[3](single_image_input)
inference_model = Model(inputs = single_image_input, outputs = embedding_output)
inference_model.set_weights(model.get_weights())

