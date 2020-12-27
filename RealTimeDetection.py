#Thiago Soares Laitz
#Huang Tzu Jan

import numpy as np
import cv2
from tensorflow import keras
from mtcnn.mtcnn import MTCNN

#Create model
model = keras.models.Sequential()
model.add(keras.layers.Input((50,50,1)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128,3,padding="same",activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(64,3,padding="same",activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(32,3,padding="same",activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(16,3,padding="same",activation="relu"))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1,activation="sigmoid"))

#Load weights trained on google Colab -- See FaceMaskDetector.ipynb
checkpoint_path = "cp.ckpt"
model.load_weights(checkpoint_path)

#Start Video Capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def limit(box):
    #Detect if the box boundaries are within Video Capture
    limit = False
    if (box[0] + box[2] <  480) and (box[1] + box[3] < 640):
        limit = True
    return limit

#Use MTCNN to detect Faces
detector=MTCNN()
while True:
    success, img = cap.read()
    face = detector.detect_faces(img)
    for face in face:
        box = face['box']
        # Crop images to 50x50 (input of neural network)
        cropped_image = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_image = cv2.resize(cropped_image,(50,50))
        cropped_image = np.array(cropped_image)
        cropped_image = cropped_image[np.newaxis,:,:,np.newaxis]

        y = model.predict(cropped_image)#Predict on image
        if(y >= 0.5):#With mask case
            if limit(box) == True:
                label = "With Mask: {:.2f}".format(y[0][0])
                cv2.putText(img, label, (box[0] - 10, box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color = (0, 255, 0))
                img = cv2.rectangle(img,
                                    (box[0], box[1]),
                                    (box[0] + box[2], box[1] + box[3]),
                                    (0, 255, 0),
                                    2)
        else:#Without mask case
            if limit(box) == True:
                label = "Without Mask: {:.2f}".format(y[0][0])
                cv2.putText(img, label, (box[0]-10, box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color=(0, 0, 255))
                img = cv2.rectangle(img,
                                    (box[0], box[1]),
                                    (box[0] + box[2], box[1] + box[3]),
                                    (0, 0, 255),
                                    2)
    #Show image
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
