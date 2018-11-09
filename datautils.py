#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:10:51 2018

@author: chen
"""
from tqdm import tqdm
from PIL import ImageFile 
from keras.preprocessing import image
import numpy as np
from sklearn.datasets import load_files 
from keras.utils import np_utils 
import glob
import os
from matplotlib import pyplot as plt
from collections import Counter

np.random.seed(42)
ImageFile.LOAD_TRUNCATED_IMAGES = True  

feature_dir = "./bottleneck_features/"
save_path = "./saved_models/"
image_path = './dogImages/'


def path_to_tensor(img_path):
    """
    Load image from a path and
    Convert it to numpy tensor
    :param img_path: path of the image file
    :return: array representation of the image
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    """
    Batch loading images
    :param img_paths: list of image file paths
    :return: image vectors as a n-d array
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

    
def load_dataset(path):
    """
    function to load train, test, and validation datasets
    :param path: root dir of train/test/valid data
    :return: list of image file name and breed label tuples
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

    
def get_breed_names(img_path="./dogImages/"):
    """
    Load breed names
    :param img_path: directory root of the dog image dataset
    :return: list of breed names
    """
    breed_names = [item[20:-1] for item in sorted(glob.glob(os.path.join(img_path, "train/*/")))]
    return breed_names


def load_face_files(path="./lfw/*/*"):
    """
    Load face image path names
    :param path: the root dir of face images
    :return: list of face image file names
    """
    human_files = np.array(glob.glob(path))
    return human_files


def show_label_distribution(path='./dogImages/train'):
    """
    Visualizae the breed label counts of any of the train, valid
     and test set
    :param path: any of the train, valid or test dir
    :return: None
    """
    _, labels = load_dataset(path)
    labels = np.argmax(labels, axis=1)
    label_counts = sorted([(l, c) for l, c in Counter(labels).items()])
    labels, counts = zip(*label_counts)
    plt.bar(labels, counts)
    plt.title("Instance Counts for Categories in the Training Set")
    plt.xlabel('Category Index')
    plt.ylabel('Counts')
    plt.show()
    
    
dog_names = get_breed_names()    
