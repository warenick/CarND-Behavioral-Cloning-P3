# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/recovery1.jpg "Recovery Image 1"
[image3]: ./examples/recovery2.jpg "Recovery Image 2"
[image4]: ./examples/recovery3.jpg "Recovery Image 3"
[video1]: ./examples/run.mp4 "Driving Video"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py code lines 57-100).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py code lines 68, 73, 78, 83,88, 93). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers(model.py code lines 89, 94, 98) and batch normalization layers in order to reduce overfitting (model.py code lines 67, 72, 77, 82, 87, 92, 97).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code TODO: line 114). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py TODO: line 144).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving on the first and the second track, recovering from the left and right sides of the road or some snake driving (moving car from side to side) on both track. Plus I used some random driving policys.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use conv layers to find features on input images and use some dense layers on the end of the network to regress steering angle.

My first step was to use a convolution neural network model similar to the LeNet from previos project. I thought this model might be appropriate because they have vonvolution layers too, but they not so big for calculating this task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% and 20%). First model usually go to the water on the first turn and I try to increase depth of the model by adding more conv layers. Finaly I used model similar to Nvidia end-to-end driving net from the paper **"End to End Learning for Self-Driving Cars"** but with some modifications. 

To combat the overfitting, I modified the model by adding batch normalization and dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-100) consisted of a convolution neural network with the following layers and layer sizes:

input_shape     -   80x160x3
cropping layer  -   45x160x3
Conv2D          -   24 filters, 5x5 kernel, 1 stride, valid padding
Conv2D          -   36 filters, 5x5 kernel, 2 strides, valid padding
Conv2D          -   48 filters, 3x3 kernel, 2 strides, valid padding
Conv2D          -   64 filters, 3x3 kernel, 1 stride, valid padding
flatten         -   (size 15680)
relu activation -   120
relu activation -   64
softmax         -   10
output layer    -   1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go to the center of the track from the side. These images show what a recovery looks like starting from left side:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had 19944 number of data points. I used left and right cam images with 0.1 bias in steering so after all I had 59832 examples.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 5 or 10 as evidenced by using keras model checkpoint. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Also you can find the model driving in the [video1](./examples/run.mp4)