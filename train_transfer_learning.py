#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:10:51 2018

@author: chen
"""
import numpy as np
import os
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from extract_bottleneck_features import feat_dict, pre_models
from datautils import (load_dataset, path_to_tensor, dog_names, feature_dir, save_path, image_path)


class TransLearner:
    """
    Class of the transfer learner
    """
    def __init__(self, name, feature_dir, save_path, image_path):
        """
        Initialization
        :param name: The pretrained model name
        :param feature_dir: path of the pretrained bottleneck features
        :param save_path: path for saving the trained models
        :param image_path: path of the dog image dataset
        """
        self.name = name
        self.feature_path = os.path.join(feature_dir, "Dog{}Data.npz".format(name))
        self.save_path = os.path.join(save_path, "weights.best.{}.hdf5".format(name))
        self.image_path_train = os.path.join(image_path, 'train')
        self.image_path_valid = os.path.join(image_path, 'valid')
        self.image_path_test = os.path.join(image_path, 'test')
        clr_input_shape = np.load(self.feature_path)['train'].shape[1:]
        self.model = self._get_model(clr_input_shape)
        self.trained = False
        
    def _get_model(self, input_shape):
        """
        Prepare the classifier head for the transfer model
        :param input_shape: the shape of the bottleneck features
        :return: model
        """
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(133, activation='softmax'))
        model.summary()
        return model
    
    def load(self):
        """
        Load trained model
        :return:
        """
        self.model.load_weights(self.save_path)
        self.trained = True
        print("Successfully loaded trained model.")
        
    def learn(self, epochs=1000, batch_size=10, lr=1e-4, lr_decay=1e-6):
        """
        Train the model
        :param epochs: number of epochs
        :param batch_size: batch size
        :param lr: learning rate
        :param lr_decay: learning rate decay rate
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
        bottleneck_features = np.load(self.feature_path)
        train = bottleneck_features['train']
        valid = bottleneck_features['valid']
        _, train_targets = load_dataset(self.image_path_train)
        _, valid_targets = load_dataset(self.image_path_valid)
        
        self.model.fit(train, 
                       train_targets,
                       validation_data=(valid, valid_targets),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[checkpointer, earlystopping],
                       verbose=1)
        self.trained = True
                  
    def test(self):
        """
        Test model on the test set
        :return: test accuracy
        """
        self.load()
        bottleneck_features = np.load(self.feature_path)
        test = bottleneck_features['test']
        _, test_targets = load_dataset(self.image_path_test) 
        predictions = [np.argmax(self.model.predict(np.expand_dims(feature, axis=0))) for feature in test]
        test_accuracy = 100*np.sum(np.array(predictions) == np.argmax(test_targets, axis=1))/len(predictions)
        print('{}, test accuracy: {:.4f}%'.format(self.name, test_accuracy))
        return test_accuracy
        
    def predict_breed(self, img_path):
        """
        Predict the breed name of the given image
        :param img_path: Image path
        :return: breed name of the image
        """
        if not self.trained:
            self.load()
        bottleneck_feature = feat_dict[self.name](path_to_tensor(img_path))
        predicted_vector = self.model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)] 

     
if __name__ == "__main__":
    test_scores = []
    for pre_m in pre_models:
        learner = TransLearner(pre_m, feature_dir, save_path, image_path)
        learner.learn(lr=5e-5)
        acc = learner.test()
        test_scores.append({"architecture":pre_m, "accuracy":acc})
        print(test_scores)
        np.save("./acc3hidden.npy", test_scores)
    print(test_scores)
             
