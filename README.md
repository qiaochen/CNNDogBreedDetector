## Project Overview
This project uses Convolutional Neural Networks (CNNs) and Transfer Learning to process real-world, user-supplied images and detect whether there is any dog or human in the images and predict the breed or the resembling breed for them respectively.
![Example Prediction](https://upload-images.jianshu.io/upload_images/3122073-6f38335d2203b480.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

This is an educational project that aims to gain practical experiences on transfer learning.

## Instructions

1.  Clone the repository and navigate to the downloaded folder.

  ```
  git clone https://github.com/qiaochen/CNNDogBreedDetector.git
  cd dog-project
  ```

2.  Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.

3.  Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.

4.  Download the bottleneck features for transfer learning
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features
-   [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
-   [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
-   [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [VGG-16](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) bottleneck features. 
Place them in the repo, at location `path/to/dog-project/bottleneck_features`.

5. Install Requirements
The core required packages are keras (on Tensorflow backend), open-cv. The detailed list can be found in the `requirements.txt` file in the `requirements` folder.
Note, that I used GPU to train the models, please refer to the [Tensorflow](https://www.tensorflow.org/install/) document to ensure the environment for GPU version is prepared.
To install all the dependencies, execute:
  ```
  pip install -r requirements/requirements-gpu.txt
  ```


6. Run The Training Code
- Non Transfer Learning model
```
python train_from_scratch.py # train the network without transfer learning
```
The trained model would be placed under the `saved_models` directory with the name `weights.best.FromScratch.hdf5`

- Transfer Learning models
```
python train_transfer_learning.py # train the transfer learning models
```
The trained models would be placed under the `saved_models` directory with the names for each base architecture.
```
weights.best.VGG16.hdf5
weights.best.VGG19.hdf5
weights.best.Resnet50.hdf5
weights.best.InceptionV3.hdf5
weights.best.Xception.hdf5
```

7. Watch Predictions
```
python breed_predictor.py
```
This code will first randomly select figures from the human face dataset and the test set of the dog image dataset, and then make predictions.
If neither dogs nor human faces are detected, no breed prediction results would be returned, otherwise, the program returns the predicted breeds (or resembling breeds for human face images) for the input images.

## Summary of Results
#### Test Performance of the Breed Classification Models
| Architecture  | Test Accuracy   |
|---|---|
|From-Scratch CNN (Initial)| 10.53|
|From-Scratch CNN (Deeper)| 16.15|
| VGG16 | 77.99 | 
| VGG19|  79.07 | 
| Resnet50| 82.30| 
|InceptionV3| 82.42|
|Xception|**85.29**|

- Deeper models achieved better performances, whether it is by transfer-learning or not.
- The best performing architecture is the transfer learning based on the Xception architecture

#### Example Results of Figure Prediction
![Example Predictions](https://upload-images.jianshu.io/upload_images/3122073-9066b1123d1f772e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## Folder Organization
```
dog-project
      |                 ## Folders ##
      |---- dogImages      # folder for the dog image dataset
      |---- haarcascades # folder for trained face detectors by OpenCV
      |---- images           # folder for image materials used in notebook
      |---- lfw                  # folder for LFW face dataset
      |---- requirements  # folder for configuring requirements
      |---- saved_models # folder for trained models
      |                  ## Files  ##
      |---- breed_predictor.py # code for predicting using trained models
      |---- datautils.py            # code for data processing
      |---- train_from_scratch.py # code for non-transfer learning
      |---- train_transfer_learning.py # code for transfer learning
      |---- extract_bottleneck_features.py # code for extract features for transfer learning
      |---- dog_app.ipynb      # notebook documenting the inital code and results
      |---- Report.md           # A detailed project report
      |---- README.md
                                    
```



## Acknowledgements
Thank all the authors and providers of the wonderful deep learning resources and software tools, which enabled me to focus only on the interesting part of the development. Thank Udacity for proving the pre-trained features and kind instructions.
