# SimpleObjectRecognition
#### Group name:A+++++               Section: 2

## Brief introduction to what we have done
Here we use the Convolutional Neural Network (CNN) and Keras to build a model to classify images (code outlined in the file `cnn-model.ipynb`). After finishing the model training and testing, a graphical user interface(GUI) is built to facilitate the application of this code trunk (code outlined in the file `GUI.ipynb`).

## Motivation 
Image classification problem is the task of assigning an input image one label from a fixed set of categories. This is one of the core problems in Computer Vision that, despite its simplicity, has a large variety of practical applications.

## What is CNN?
A CNN is a supervised learning technique which needs both input data and target output data to be supplied. These are classified by using their labels in order to provide a learned model for future data analysis.

Typically a CNN has **three main constituents** - a **Convolutional Layer**, a **Pooling Layer** and a **Fully connected Dense Network**. The Convolutional layer takes the input image and applies m number of nxn filters to receive a feature map. The feature map is next fed into the max pool layer which is essentially used for dimensionality reduction, it picks only the best features from the feature map. Finally, all the features are flattened and sent as input to the fully connected dense neural network which learns the weights using backpropagation and provides the classification output.

CNNs use **relatively little pre-processing** compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.

[readmore](https://en.wikipedia.org/wiki/Convolutional_neural_network)  

## Data
Data in this project is devided into **three sections**. 
* **Training data**: the data we use to build the models
* **Testing data**: untouched data to verify the effectiveness of the built model
* **Online data**: evaluate the CNN model from random images searched online and do some interesting application

## Implementation and steps

### Image Data download and extraction(`extract-cifar10.ipynb`)
* extract the images from their labels and extract the ten categories
* construct the training and testing dataset
    * training dataset: 5 batches data ,each of which contains 10000 images
    * testing dataset: 1 batch data, that is 10000 images 
* This program is meant to be run once for all. <br\>
In case that you accidentally run the program twice and to avoid duplicating images, we wrote a  'refresh_folder' function to make sure that you have deleted everything before saving the images and labels.
* save the images and labels and converte the images into ‘png’ format

### Image Data processing 
* construct a dictionary with image name and label in file `cnn-model.ipynb`
* use the one hot encoding to change each label to an array with 1 and 0. <br\>
This process will accelerate our training time. 
* reshape the image data to a 1-d array with a length of 32*32

### Modeling
* **Convolutional Layer #1**: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
* **Maximize Pooling Layer #1**: Performs max pooling with a 5x5 filter and stride of 1 (which specifies that pooled regions do not overlap)
* **Convolutional Layer #2**: Applies 50 5x5 filters, with ReLU activation function
* **Maximize Pooling Layer #2**: Again, performs max pooling with a 5x5 filter and stride of 1
* **Convolutional Layer #3**: Applies 80 5x5 filters, with ReLU activation function
* **Maximize Pooling Layer #3**: Again, performs max pooling with a 5x5 filter and stride of 1
* **Dense Layer #1**: 512 neurons, with dropout regularization rate of 0.25
* **Dense Layer #2 (Logits Layer)**: 10 neurons, one for each digit target class (0–9).

<img src="https://github.com/nding17/SimpleObjectRecognition/blob/master/data/readme_data/pic4.png" width="300" height="500">

## Results
* when we evaluate the model based on training data, the accuracy rate is 78.65%, when we evaluate the model based on testing data, he accuracy rate is 61.66. We think this result is fine becauese of there are  total 10 categories.

## Installation instructions
In order to run the code above, you need to install some essential liabraries:<br/>
```pip install TensorFlow  ``` <br/>
```pip install keras ```


## Run instructions
To use our train model, only need to run `GUI.ipynb` file and the weights of our model saved in file `model.h5`.
* To run `GUI.ipynb`: Open the file and run the file. An friendly interface will appear and direct you to choose a picture as follows.
    * upload image window 
    
    <img src="https://github.com/nding17/SimpleObjectRecognition/blob/master/data/readme_data/pic1.jpg" width="300" height="300">
    
    * choose an image from your own desktop
    
    <img src="https://github.com/nding17/SimpleObjectRecognition/blob/master/data/readme_data/pic2.jpg" width="300" height="300">
    
    * return the recognization result
     
    <img src="https://github.com/nding17/SimpleObjectRecognition/blob/master/data/readme_data/pic3.jpg" width="300" height="300">
