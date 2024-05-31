# Face Recognition System

Objective of this project is to develop facial recognition models which utilize deep convolutional neural networks with transfer learning, one-shot learning methods, and Siamese networks for efficient and accurate face recognition. 
This project enabled the team to understand how Neural networks are trained both mathematically as well as in implementation, the concept of feature extraction using convolution as well as the basics of tensorflow and keras. It enabled the
developers to understand the fundamentals of computer vision and develop models that work on Transfer Learning, One-Shot Learning and Concept of similarity for face identification. 

The project was divided into 3 components:
1) FaceNet Approach
2) Siamese Network Approach
3) Haarcascade Classifier Approach

We thank Dr. Shashank Srivastava and Dr. Shailendra Shukla (Professors CSED, MNNIT Allahabad) under whose guidance this project was developed!

# FaceNet
We tried to implement the ideas given in the original FaceNet paper and our initial idea was to the InceptionV3 model by Google as the base dense architecture and then build upon that by excluding top of InceptionV3
and adding a Global Average Pooling followed by a dropout layer, a fully connected layer of 128 neurons to bring down the size to 128 Dimentions as defined in the original FaceNet paper followed by BatchNorm layer,
this was the most successful combination achieved by us, the plan was to train the entire InceptionV3 along with the three layers, utilizing offline triplet mining through a data generator and a custom triplet loss function to 
train the model. 

We couldn't train the entire model on the CASIA WebFace dataset due to lack of compute, and that leaves a further project motivation, to test out our best pipeline using hopefully an online triplet generation and CASIA Webface dataset by training it utilizing the GPU available at [MNNIT Allahabad](https://mnnit.ac.in/) in months to come.
We did test out our concept and using around 7500 images from LFW dataset and training only the extra layers added, we achieved an accuracy of around 35% on the never seen by the model dataset of BLFW

Then we decided to use a pre-trained model and somehow try to make it perform better on Indian Faces, we chose the David Sandberg's FaceNet implementation ([Link to his repo](https://github.com/davidsandberg/facenet)) and fine tuned it using the Indian Actor Faces Dataset
This improved the accuracy of the model by 6% for Indian Faces, the model gives an accuracy of 100% on a class size of 20 and an accuracy of 75% on class size of 1600 classes, however we used KNN for classification once embeddings were generated and accuracy is subjuctive to agorithm used for classification

Here's link to the best outcome model of this part of this project:
[FaceNet_TL_ep10](https://drive.google.com/file/d/1nEcURXNSUZYv8BsESmtLbLAmy0xIMHyR/view?usp=sharing)

We thank David Sandberg and Nyoki for their immense contribution to the open source community and providng us the pre-trained model!


 # Contributers
 1) [Ayushman Tiwari](https://github.com/Ayushman444)
 2) [Aviral Mishra](https://github.com/aviral-mishra-1008)
 3) [Vaibhav Sharma](https://github.com/IWantToBeVS)
 4) [Shreyansh Karamtot](https://github.com/Fire-guy)

 # Technology Used/References
 The project was made using the python language and thus you can refer to [python.org](https://www.python.org/) for more information.
 Following Libraries were used primarily in the development of the project:

 1) Tensorflow (version 2.15.0) [Considering the fleeting nature of tensorflow updates, we'd recommend using tensorflow 2.15.0 for running this project]
 2) OpenCV
 3) Keras (now integrated into tensorflow itself)
 4) Haarcascade Frontel Face XML file was used in face detection and a wide variety of haarcascade can be found [here](https://github.com/opencv/opencv/tree/4.x/data/haarcascades)

Following research papers were referenced for a better understanding and development of the project:
1. [FaceNet:  A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
2. [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
3. [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

# Datasets Used
Following datasets were used primarly for training the initial models of FaceNet and Siamese Network:
1. [Labelled Faces In Wild(LFW)](https://vis-www.cs.umass.edu/lfw/) -> Primarily Used For Training Siamese Network Model and Initial Pipelines of FaceNet
2. [CASIA WebFace](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface/discussion) -> Used For training the pre-trained pipeline model used in FaceNet as well as tried to train the second model proposition using this dataset
3. [Indian Actors Dataset](https://www.kaggle.com/datasets/nagasai524/indian-actor-faces-for-face-recognition) -> Used to fine tune the pre-trained model pipeline for better acccuracy on Indian Faces
4. [Balanced Faces In Wild (BLFW)](https://ieee-dataport.org/documents/balanced-faces-wild) -> This was used primarly as the testing dataset for FaceNet and its Indian Subset was used for testing the accuracy of pre-trained model as well as the fine tuned model for Indian Faces
5. [IISCIFD](https://github.com/harish2006/IISCIFD) -> Used to test the accuracy of FaceNet model over 1600 classes

We thank all data providers for providing such useful datasets for free for research and education purposes!

Collaborate, Discus and Feel Free To Share Our Work



