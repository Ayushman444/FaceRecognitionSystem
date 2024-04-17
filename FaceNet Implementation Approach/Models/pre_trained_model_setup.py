'''THIS CODE IS TO DOWNLOAD NECESSARY ITEMS'''
# from google.colab import drive
# drive.mount('/content/drive')

# !cp '/content/drive/MyDrive/FaceNet David/facenet_keras_weights.h5' /content
# drive.flush_and_unmount()

'''THIS MODEL REQUIRES THE model_architecture file and the facenet_keras_weights.h5 file to setup completely, credits to Gauravesh for the awesome repository and training weights available on repo were used on our project'''
#https://github.com/gauravesh/Face-recognition-Using-Facenet-On-Tensorflow-in_colab LINK TO ORIGINAL MODEL WEIGHTS AND ARCHITECTURE REPO


class FaceNet:
  def __new__(self,summary=False):
    from model_architecture import InceptionResNetV2
    facenet = InceptionResNetV2()
    facenet.load_weights('/content/facenet_keras_weights.h5')
    if summary:
      print(facenet.summary())
    return facenet
