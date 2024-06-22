import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
import cv2
from Models.FaceNet_third_model_proposition_using_preTrained_fineTuning import *
from sklearn.preprocessing import Normalizer

# Load your FaceNet model and face embeddings
l2_normalizer = Normalizer('l2')
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = Embed_model()

with open("faceEmbeddings.pkl", "rb") as f:
    data = pickle.load(f)

class_name_and_arrays = data

# Open the webcam (camera index 0)
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        # Get the face embedding
        face_embedding = model.predict(face_img)[0]
        face_embedding = l2_normalizer.transform(np.expand_dims(face_embedding, axis=0))[0]

        # Find the most similar class (person's name)
        highest_similarity = -float('inf')
        predicted_class_name = None
        for array, class_name in class_name_and_arrays:
            similarity = np.dot(face_embedding, array) / (np.linalg.norm(face_embedding) + np.linalg.norm(array))
            if similarity > highest_similarity:
                highest_similarity = similarity
                predicted_class_name = class_name

        # Draw a bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add the predicted name as text
        cv2.putText(frame, predicted_class_name.upper(), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)


    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
