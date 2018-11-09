#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:10:51 2018

@author: chen
"""
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint  
from datautils import (load_dataset, paths_to_tensor, path_to_tensor, dog_names, save_path, image_path)
import os


class CNNClassifier:
    """
    The non-transfer learner
    """
    def __init__(self, 
                 name, 
                 save_path, 
                 image_path):
        """
        Initialization
        :param name: name of the learner
        :param save_path: path for saving trained model
        :param image_path: path of the dog image dataset
        """
        self.name = name
        self.save_path = os.path.join(save_path, "weights.best.{}.hdf5".format(name))
        self.image_path_train = os.path.join(image_path, 'train')
        self.image_path_valid = os.path.join(image_path, 'valid')
        self.image_path_test = os.path.join(image_path, 'test')
        self.model = self._get_model()
        self.trained = False
        
    def _get_model(self):
        """
        Construct model architecture
        :return: model
        """
        model = Sequential()
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', input_shape=(224, 224, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.15))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(133, activation='softmax'))
        model.summary()
        return model
        
    def learn(self, epochs=8000, batch_size=10, lr=1e-4, lr_decay=1e-5):
        """
        Traing the model
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param lr: Learning rate
        :param lr_decay: Learning rate decay rate
        :return:
        """
        earlystopping = EarlyStopping(monitor='val_loss', 
                                      min_delta=0, 
                                      patience=15, 
                                      verbose=1, 
                                      mode='auto', 
                                      baseline=None,
                                      restore_best_weights=True)
                                      
        checkpointer = ModelCheckpoint(filepath=self.save_path, 
                                       verbose=1, 
                                       save_best_only=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=lr, decay=lr_decay),
                           metrics=['accuracy'])
        train_files, train_targets = load_dataset(self.image_path_train)
        valid_files, valid_targets = load_dataset(self.image_path_valid)
        
        # pre-process the data for Keras
        train_tensors = paths_to_tensor(train_files).astype('float32')/255
        valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
        
        self.model.fit(train_tensors, 
                       train_targets,
                       validation_data=(valid_tensors, valid_targets),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[checkpointer, earlystopping],
                       verbose=1)
        self.trained = True
    
    def load(self):
        """
        Load trained model
        :return:
        """
        self.model.load_weights(self.save_path)
        self.trained = True
                  
    def test(self):
        """
        Test model in the test set
        :return: test accuracy
        """
        self.load()
        test_files, test_targets = load_dataset(self.image_path_test) 
        test_tensors = paths_to_tensor(test_files).astype('float32')/255
        predictions = [np.argmax(self.model.predict(np.expand_dims(feature, axis=0))) for feature in test_tensors]
        test_accuracy = 100*np.sum(np.array(predictions) == np.argmax(test_targets, axis=1))/len(predictions)
        print('{}, test accuracy: {:.4f}%'.format(self.name, test_accuracy))
        return test_accuracy
        
    def predict_breed(self, img_path):
        """
        Predict breed given an input image
        :param img_path: image path
        :return:predicted breed name
        """
        if not self.trained:
            self.load()
        img_tensor = path_to_tensor(img_path)
        predicted_vector = self.model.predict(img_tensor)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]
        
if __name__ == "__main__":
    clr = CNNClassifier("FromScratch", save_path, image_path) 
    clr.learn()
    clr.test()     
     
