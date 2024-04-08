import os
import cv2
import numpy as np

def calculate_accuracy(test_data_dir, clf):
    total_faces = 0
    correct_predictions = 0

    for filename in os.listdir(test_data_dir):
        img_path = os.path.join(test_data_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        ground_truth_id = int(filename.split('.')[1])

        predicted_id, _ = clf.predict(img)

        if predicted_id == ground_truth_id:
            correct_predictions += 1

        total_faces += 1

    accuracy = (correct_predictions / total_faces) * 100
    print("Accuracy: {:.2f}%".format(accuracy))

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("C:/Users/Lenovo/Desktop/FaceRecognitionSystem/classifier.xml")

test_data_dir = "C:/Users/Lenovo/Desktop/FaceRecognitionSystem/SampleData"


calculate_accuracy(test_data_dir, clf)
