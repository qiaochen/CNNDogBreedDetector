## Project Definition

### Project Overview
This project uses Convolutional Neural Networks (CNNs) and Transfer Learning to process real-world, user-supplied images and detect whether there is any dog or human in the images and predict the breed or the resembling breed respectively.

#### Dataset
- [Dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) which has been split into training, validation and testing sets. There are 133 total dog categories and 8351 total dog images in the dataset. The training, validation and testing sets contain 6680, 835 and 836 dog images respectively. This dataset is used to train and test the dog breed detector.

- [Human face dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip), contains 13233 total human images. (Pretrained face detector is used in this project, the dataset is mainly used for validation purpose)

#### Pretrained Models
-  OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) is used to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades). For the first 100 dog images in the `test dog dataset` and the first 100 face images of the `human face dataset`, it predicts 12% and 100% of them to have human faces respectively.

- [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006), 
directly used as the dog detector in this project, imported using keras api. For the first 100 dog images in the `test dog dataset` and the first 100 face images of the `human face dataset`, it predicts 100% and 1% of them to have dogs respectively. 


- candidate architectures for transfer learning of the dog breed classifier.
  - [VGG-16](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) bottleneck features
  - [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
  - [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
  - [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
  - [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

### Problem Statement
Given an image, the classifier will detect whether there is a dog. If there is a dog, the trained neural classifier will identify an estimate of the canine’s breed. If the supplied image is an image of a human, the classifier will identify the resembling dog breed.
Below are examples of the detection and prediction results.

![Detected a Dog and Predict Its Breed](https://upload-images.jianshu.io/upload_images/3122073-7a3fdee5db3f0f24.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![Detected Human Face and Predict the Resembling Dog Breed](https://upload-images.jianshu.io/upload_images/3122073-486d60aa385334cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Metrics
1. Validation Performance
Accuracy is used to measure the performance of the classifiers, based on the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), accuracy is computed as follows:
```
accuracy = (TP + TN) / (TP + TN + FP + FN) 
```
The score range is [0, 1], with the 0 the worst while 1 the best.
Accuracy is adopted because the target variable classes for most categories in the data are nearly balanced, and the target variable classes in the data are not a majority of one class. In this setting, accuracy is comparable to f1 score and is better than precision and recall which do not consider True Negatives. 

2. Training Loss
[Categorical cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) is used as the loss function for computing gradients for network optimization. The loss is based on the following equation.

![Cross-Entropy Loss](https://upload-images.jianshu.io/upload_images/3122073-0d4e31590dd61361.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Because the classification task is a multi-class prediction task, cross-entropy is adopted as the objective. From the equation above, we can know that cross-entropy loss rewards/ penalizes probabilities of correct classes only. The value is independent of how the remaining probability is split between incorrect classes.
---
## Analysis
Let us focus on the dog breed dataset, on which I will train and validate the dog breed classifier. As has been mentioned above, there are 133 total dog categories and 8351 total dog images in the dataset. The training, validation and testing sets contain 6680, 835 and 836 dog images respectively.
### Image Data Exploration and Visualization
- The original dog images are not normalized. There can be images in arbitrary sizes, such as 400*300, 480*360, 321*316 (The first three images in the training set).
- The color images have three color channels.
- The following are the positive sample counts in the training set.
It should be mentioned that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imbalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.
![Traning Set Counts](https://upload-images.jianshu.io/upload_images/3122073-268180b9db28da8f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---
## Methodology
###  Data Preprocessing
When using TensorFlow as backend, Keras CNNs require a 4D array (which is also referred to as a 4D tensor) as input, with shape
```
(nb_samples,rows,columns,channels),
 ```
where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.

The `path_to_tensor` function in the `datautils.py` and the ipython notebook takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN. The function first loads the image and resizes it to a square image that is  224×224  pixels. Next, the image is converted to an array, which is then resized to a 4D tensor. In this case, since we are working with color images, each image has three channels. Likewise, since we are processing a single image (or sample), the returned tensor will always have shape
```
(1,224,224,3).
 ```
For batch processing, the `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape
```
(nb_samples,224,224,3).
 ```
Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths. It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in the dataset.
### Implementation
#### 1. A CNN from scratch
##### 1.1 Architecture
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 223, 223, 64)      832       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 111, 111, 64)      0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 111, 111, 64)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 110, 110, 128)     32896     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 55, 55, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 55, 55, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 54, 54, 128)       65664     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 27, 27, 128)       0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 128)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               33024     
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 133)               34181     
=================================================================
Total params: 166,597
Trainable params: 166,597
Non-trainable params: 0
_________________________________________________________________
```
- The CNN layers extract high-level semantic feature by working on lower-level features.
- Dropout layers randomly drop part of the network, so that each part of the neural networks can have the opportunity of getting tunned alone.
- Relu activation function ensures non-linear transformations while preventing gradient saturation.
- Deeper models are known to enbale better performance, so I added two fully-connected layers before the output softmax layer.


##### 1.2 Strategies and Hyperparameters
  - Loss: Categorical Cross Entropy
  - Metrics: Accuracy
  - Optimizer: Adam
  - Learning Rate: 1e-2
  - Learning Rate Decay: 1e-6
  - Number of epoch: 2000
  - Early-Stopping:
    - Minimum performance improvement: 0 
    - Patience: 15
    - Restore from best model if the performance does not improve

Here a larger learning rate at the beginning can faster training and may enable jumping out of local minimums.
Schedule the learning rate to decay epoch after epoch makes later gradient update favor fine-grained tunning so that the convergence can be more stable.
An early stopping scheme is scheduled to prevent the model from fitting overly well on the training set whereas generalizing poorly.

#### 2. Transfer learning
##### 2.1 Architecture
The network architecture for transfer learning is illustrated below, the extracted features from the pre-trained network are pooled and flatterned using the `global_average_pooling2d_2` layer going through the fully-connected hidden and output layers.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_2 ( (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               131328    
_________________________________________________________________
dropout_5 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               32896     
_________________________________________________________________
dropout_6 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 133)               17157     
=================================================================
Total params: 181,381
Trainable params: 181,381
Non-trainable params: 0
_________________________________________________________________

```
- Add Dropout layer to make each part of the network have the chance to be isolatedly trained so that the network performance can be more generalizable.
- Make deeper architecture to get better performance.
- Use Relu activation function to avoid gradient saturation.

##### 2.2 Strategies and Hyperparameters
  - Loss: Categorical Cross Entropy
  - Metrics: Accuracy
  - Optimizer: Adam
  - Learning Rate: 1e-2
  - Learning Rate Decay: 1e-6
  - Number of epochs: 200
  - Early-Stopping:
    - Minimum performance improvement: 0 
    - Patience: 15
    - Restore from the best model if the performance does not improve

Here a larger learning rate at the beginning can faster training and may enable jumping out of local minimums.
Schedule the learning rate to decay epoch after epoch makes later gradient update favor fine-grained tunning so that the convergence can be more stable.
An early stopping scheme is scheduled to prevent the model from fitting overly well on the training set whereas generalizing poorly.

### 2. Refinement
- Refinements were made to hyper-parameters of both model types. The learning rates of both architectures were refined to be smaller (5e-5, and 1e-4 for non-transfer and transfer learning respectively). These refinements were made as the initial trials showed that the learning rates were too high to enable robust loss decreases.
- Besides, one more CNN layer and one more fully-connected hidden layer were added to the "from-scratch" architecture, as initial trials indicated increased performance with more layers. 
The new "from-scratch" architecture is summarized as follows:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 223, 223, 128)     1664      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 111, 111, 128)     0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 111, 111, 128)     0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 110, 110, 128)     65664     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 55, 55, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 55, 55, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 54, 54, 128)       65664     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 27, 27, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 26, 26, 128)       65664     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 13, 13, 128)       0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 128)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               33024     
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896     
_________________________________________________________________
dropout_5 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 128)               16512     
_________________________________________________________________
dropout_6 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 133)               17157     
=================================================================
Total params: 298,245
Trainable params: 298,245
Non-trainable params: 0
_________________________________________________________________
```
- However, a trial on adding additional hidden layers indicated no significant improvements for the transfer learning architecture. As a result, it is kept in the initial form.

---
## Results
### Model Evaluation and Validation
| Architecture  | Test Accuracy   |
|---|---|
|From-Scratch CNN (Initial)| 10.53|
|From-Scratch CNN (Refined)| 16.15|
| VGG16 | 77.99 | 
| VGG19|  79.07 | 
| Resnet50| 82.30| 
|InceptionV3| 82.42|
|Xception|**85.29**|

The transfer learning architectures share the same top classifier layers and the training strategy/parameters. In this condition, the best performing model is the one with the Xception bottleneck features. 

Below are the prediction results of the best model on 5 randomly sampled images from the test dataset.
![Prediction Results](https://upload-images.jianshu.io/upload_images/3122073-ce106f577dc429cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### Discussion and Justification
In general, better performances were achieved by the "deeper" models.
- The refined "from-scratch" CNN has one more convolutional layer and one more fully-connected layer than the initial version, although requiring much more training time, the cost is worthwhile and it obtained a great performance gain. The reason could be that with deeper layers, more complex semantics can be learned by the networks.
-  The transfer-learning architectures VGG16 and VGG19 (with 16 and 19 network layers respectively) also exposed the same trends: Deeper model gains better performance (77.99 vs 79.07). ResNet50, which is much deeper than both VGG16 and VGG19, obtained even better performance (82.30).
- Besides solely relying on the depth of the network architecture, special designs of sub-structures like the "residual modules" in ResNet50, "Inception micro-architecture" in InceptionV3, and "depthwise separable convolutions." in Xception also benefited the overall model performance.

---
## Conclusion
In this project, I combined human face and dog detectors with a neural dog breed classifier, such that given an image from the input end, the algorithm will detect human faces or dogs and predict the breeds (resembling breed for human faces). The output end is the predicted target type with the breed label.

I leveraged pre-trained human face and dog detectors while focused on the training of breed classifiers. I tried to tackle the breed classification problem by constructing a CNN architecture from scratch, as well as applying transfer learning on top of those famous pre-trained architectures. 

The best model was obtained using pre-trained ` {}` as feature extractor appended with two additional hidden fully-connected layers and an output softmax layer. After processing the input image, the breed type with the maximum probability produced from the softmax layer will be assigned as the breed of the target in the image.

### Reflection

- Training the classifier from scratch was quite challenging, yet the performance is far behind the classifiers trained with transfer learning. Accessible training data, scale, and computational resources are the bottlenecks that stop many ideas of trying. However, the advantage of adding layers to make the network deeper was observed when the validation accuracy score increased with the number of hidden layers.

- It was also observed that too large a learning rate disabled the learning of the networks. The experience in this project is to set the learning rate of Adam to be around 1e-4 and decay the learning rate episodically. The initial trial of 1e-2 for the learning rate showed no trends of learning at all.


### Possible Improvement Directions
- Train the networks using data augmentation, so that more invariant features could be captured by the networks and slight transformations of the same image would not affect the judgment of the classifier.
- Add adversarial mechanism in the network training procedure, to make the detector stronger in distinguishing true targets from similar while negative images.
- Ensemble a set of detectors to predict based on collective intelligence.


