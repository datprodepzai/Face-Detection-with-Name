import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


train_path = 'dataset'
test_path = 'test_images'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids



def testAllFaces(test_path):
    results = []
    imagePaths = [os.path.join(test_path, f) for f in os.listdir(test_path)]

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(img_numpy[y:y + h, x:x + w])
            results.append((id, confidence))

    return results


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(train_path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')


num_faces = len(np.unique(ids))
num_images = len(ids)
print("\n [INFO] {0} faces trained from {1} images. Exiting Program".format(num_faces, num_images))


results = testAllFaces(test_path)


plt.figure(figsize=(8, 6))
labels = ['Trained Faces', 'Total Images', 'Tested Faces']
values = [num_faces, num_images, len(results)]
plt.bar(labels, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Training and Testing Summary')
plt.ylabel('Count')
plt.grid(axis='y')
plt.savefig('trained_and_tested_faces_summary_chart.png')
plt.show()
