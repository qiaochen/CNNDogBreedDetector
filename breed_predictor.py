#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:10:51 2018

@author: chen
"""
from keras.applications.resnet50 import (preprocess_input, ResNet50)
from datautils import path_to_tensor
import numpy as np
import cv2
import os
from train_transfer_learning import TransLearner
from datautils import (feature_dir, save_path, image_path, load_dataset, load_face_files, dog_names)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 


class BreedPredictor:
    """
    Class wrapping the image prediction function
    """
    
    def __init__(self, 
                 model_name="Xception", 
                 cascade_path='haarcascades/haarcascade_frontalface_alt.xml'):
        """
        Initialization
        :param model_name: The transfer-learning architecture to load
        :param cascade_path: The path of the trained face detector
        """
        self.resNet50_model = ResNet50(weights='imagenet')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.breed_predictor = TransLearner(model_name, feature_dir, save_path, image_path)

    def _resNet50_predict_labels(self, img_path):
        """
        returns prediction vector for image located at img_path
        :param img_path: path of an image file
        :return: the label of the image
        """
        _img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(self.resNet50_model.predict(_img))

    def face_detector(self, img_path):
        """
        Detect faces
        :param img_path: path of an input image
        :return: True if a face is detected, False otherwise
        """
        _img = cv2.imread(img_path)
        gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def dog_detector(self, img_path):
        """
        Detect dogs
        :param img_path: image path
        :return: True if a dog is detected, False otherwise
        """
        prediction = self._resNet50_predict_labels(img_path)
        return (prediction <= 268) & (prediction >= 151)

    def predict_breed(self, img_file):
        """
        Predict breed
        :param img_file: image path
        :return:  detection result
        """
        is_dog = self.dog_detector(img_file)
        if is_dog:
            return "dog", self.breed_predictor.predict_breed(img_file)
        
        is_human = self.face_detector(img_file)
        if is_human:
            return "human", self.breed_predictor.predict_breed(img_file)
        return None

    
if __name__ == "__main__": 
    n_test = 6
    predictor = BreedPredictor()
    dog_files, dog_labels = load_dataset(os.path.join(image_path, 'test'))
    dog_labels = np.argmax(dog_labels, axis=1)
    human_files = load_face_files()
    
    dog_file_indices = np.random.choice(range(len(dog_files)), n_test//2, replace=False)
    dog_files = [(dog_files[idx], dog_labels[idx]) for idx in dog_file_indices]
    face_files = np.random.choice(human_files, n_test // 2, replace=False)
    face_files = [(ff, -1) for ff in face_files]
    
    image_files = dog_files + face_files
    images = []
    res_strs = []
    for image_f, label in image_files:
        res = predictor.predict_breed(image_f)
        img = mpimg.imread(image_f)
        images.append(img)
        if res is None:
            print("Neither dogs nor humans found!")
            res_strs.append("Neither dogs nor humans found!")
            continue
        if res[0] == "human":
            print("Hello human,\nyou look like a {}".format(res[1]))
            res_strs.append("Hello human,\nyou look like a {}".format(res[1]))
        else:
            print("Hey, this is a dog,\nthe breed is {}\n(Ground-True breed: {}).".format(res[1], dog_names[label]))
            res_strs.append("Hey, this is a dog,\nthe breed is {}\n(Ground-True breed: {}).".format(res[1], dog_names[label]))
        print("=="*10)
    for idx, img in enumerate(images):
        plt.subplot(len(images),2, 2*idx+1, frameon=False)
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.text(0.5, 0.5, res_strs[idx], horizontalalignment='center', verticalalignment='center')
        plt.subplot(len(images),2, 2*idx+2, frameon=False)
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
    plt.savefig('prediction_res.png')
    plt.show()
    
