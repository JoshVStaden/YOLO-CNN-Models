import os, time, random, argparse
import numpy as np
import tensorflow.keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

class YOLO:
    def VGG16_body(self, input_shape, num_classes):
        vgg = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
        vgg.trainable = False

        model = Sequential([
            vgg,
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=256, activation='relu'),
            Dense(units=2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        return model
        


    def get_Yolo3_model(self,num_classes):
        return self.VGG16_body((224, 224, 3), num_classes)



    def __init__(self, model_type):
        self.model = self.get_Yolo3_model(2)
        self.hist = None

    def train_model(self, train_dir, test_dir):
        # Preprocess images
        trdata = ImageDataGenerator()
        traindata = trdata.flow_from_directory(directory=train_dir, target_size=(224,224))
        tsdata = ImageDataGenerator()
        testdata = tsdata.flow_from_directory(directory=test_dir, target_size=(224,224))
        self.hist = self.model.fit(traindata, steps_per_epoch=70, validation_data=testdata, validation_steps=10, epochs=5)
    
    def save_model(self, filename="latest_model.h5"):
        self.model.save(filename)
    
    def visualize_model(self):
        if self.hist is None:
            print("No history, cannot visualize")
        else:
            plt.plot(self.hist.history(["accuracy"]))
            plt.plot(self.hist.history(["val_accuracy"]))
            plt.plot(self.hist.history(["loss"]))
            plt.plot(self.hist.history(["val_loss"]))
            plt.title("model accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.legend(["Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])
            plt.show()

yolo = YOLO("yolo")
yolo.train_model("model_images/train_set", "model_images/test_set")