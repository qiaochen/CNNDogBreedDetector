#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:10:51 2018

@author: chen
"""


def extract_VGG16(tensor):
    """
    Extract VGG16 bottleneck features
    :param tensor: Image input tensor
    :return: VGG16 features
    """
    from keras.applications.vgg16 import VGG16, preprocess_input
    return VGG16(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def extract_VGG19(tensor):
    """
    Extract VGG19 bottleneck features
    :param tensor: Image input tensor
    :return: VGG19 features
    """
    from keras.applications.vgg19 import VGG19, preprocess_input
    return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def extract_Resnet50(tensor):
    """
    Extract Resnet50 bottleneck features
    :param tensor: Image input tensor
    :return: Resnet50 features
    """
    from keras.applications.resnet50 import ResNet50, preprocess_input
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def extract_Xception(tensor):
    """
    Extract Xception bottleneck features
    :param tensor: Image input tensor
    :return: Xception features
    """
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def extract_InceptionV3(tensor):
    """
    Extract InceptionV3 bottleneck features
    :param tensor: Image input tensor
    :return: InceptionV3 features
    """
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


feat_dict = {"VGG16": extract_VGG16,
             "VGG19": extract_VGG19,
             "Resnet50": extract_Resnet50,
             "InceptionV3": extract_InceptionV3,
             "Xception": extract_Xception}

pre_models = ["VGG16", "VGG19", "Resnet50", "InceptionV3", "Xception"]  
