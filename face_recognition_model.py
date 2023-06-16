import pickle

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D, MaxPool2D, Flatten, Dropout
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
import seaborn as sns
import numpy as np
import sys
#import cv2 as cv
from matplotlib.image import imread
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

train_data_path = 'Data_set/train'
test_data_path = 'Data_set/val'


data_gen = ImageDataGenerator(
                               rescale=1/ 255,
                                rotation_range=10,
                                zoom_range=0.3,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                #fill_mode='constant',
                                horizontal_flip=True
                              )
train_set = data_gen.flow_from_directory(
                                        train_data_path,
                                        batch_size= 64,
                                        color_mode='rgb',
                                        shuffle=True,
                                        class_mode= 'sparse'
                                        )


test_set = data_gen.flow_from_directory(
                                        test_data_path,
                                        batch_size=64,
                                        color_mode='rgb',
                                         shuffle=True,
                                        class_mode= 'sparse'
                                        )

train_set_length = len(train_set.class_indices)


model = Sequential()

dimension = (256, 256, 3)

# CONVOLUTIONAL LAYER 1
model.add(Conv2D( filters=64, kernel_size=(4, 4), input_shape=dimension, activation='relu'))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# CONVOLUTIONAL LAYER 1
model.add(Conv2D( filters=256, kernel_size=(4, 4), activation='relu'))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))


#CONVOLUTIONAL LAYER 2
model.add(Conv2D(filters=128, kernel_size=(4,4), activation='relu'))
#POOLING LAYER 2
model.add(MaxPool2D(pool_size=(2,2)))


#CONVOLUTIONAL LAYER 2
model.add(Conv2D(filters=256, kernel_size=(4,4), activation='relu'))
#POOLING LAYER 2
model.add(MaxPool2D(pool_size=(2,2)))


# FLATTEN IMAGES FROM 224 by 224 * 3 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# LAST LAYER IS THE CLASSIFIER
model.add(Dense(train_set_length, activation='softmax'))

# COMPILE THE MODEL
model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',
             metrics = ['accuracy'])

model.summary()

classes = { train_set.class_indices[key]: key for key in train_set.class_indices}

pickle.dump(classes,open('classes.p','wb'))
# results =
def train(trainSet, testSet, epochs) :
    # early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=10)
    model.fit( trainSet, epochs=epochs, validation_data=testSet)

def saveModel(model, saveAs):
    return model.save(saveAs)

def convertImgToArray(img):
    newImg = []
    if type(img) == str:
        myImage = str(img)
        TheIMG = image.load_img(myImage,target_size=(256,256))
        newImg = image.img_to_array(TheIMG)
        newImg = np.expand_dims(newImg,axis=0)
        newImg = newImg/255
    return newImg

def take_attendance(img):
    newImg = []
    if type(img) == str:
        newImg = convertImgToArray(img)


    newImg =  np.expand_dims(img,axis=0)
    newImg = newImg/255
    myModel = load_model('face_recognition_model.keras', compile=False)
    pred_probabilities = myModel.predict(newImg)

    print(pred_probabilities)
    predictions = pred_probabilities > 0.9
    print(predictions)
    prediction = [i for i, x in enumerate(predictions[0]) if x]
    print(prediction)
    if prediction:
        return classes[prediction[0]]
    return "Not Found"

def train(trainSet, testSet, epochs) :
    model.fit( trainSet, epochs=epochs, validation_data=testSet)

if __name__ == "__main__":
    # myImage = train_data_path+'/170805504/001_9adc92c2.jpg'
    # # result = take_attendance(myImage)

    # # print(result)
    # # predictions = model.predict(test_set)
    # # test_set = tf.reshape(test_set, [256, 256])
    # # train_set = tf.reshape(train_set, [256, 256])
    # train(train_set, test_set, 2)
    # saveModel(model, 'face_recognition_model.keras')
    # # print(classes)


    train(train_set, test_set, 100)
    saveModel(model, 'face_recognition_model.keras')

