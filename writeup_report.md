# **Behavioural Cloning** 

## Report

---

**Behavioural Cloning Project**


[//]: # (Image References)

[25epDropout]:  ./reportPics/nvidiaDropoutTraining25Epochs.png "Training with drop out"
[25epNvid]: ./reportPics/nvidiaTraining25Epochs.png "Training without drop out"
[centerDriving]: ./reportPics/center_2018_01_11_19_51_28_257.jpg "Example of center driving"
[centerDrivingOpp]: ./reportPics/center_2018_01_11_19_56_06_394.jpg "Example of center driving in opposite direction"
[recoveryRight]: ./reportPics/center_2018_01_13_21_12_05_988.jpg "Example of Recovery from right"
[recoveryLeft]: ./reportPics/center_2018_01_13_21_12_24_650.jpg "Example of Recovery from left"
[centerDriving1]: ./reportPics/center_2018_01_11_20_21_18_106.jpg "Example of center driving on right track"
[centerDrivingOpp1]: ./reportPics/center_2018_01_11_20_42_56_162.jpg "Example of center driving in opposite direction right track"
[recoveryRight1]: ./reportPics/center_2018_01_15_12_04_07_229.jpg "Example of Recovery from right right track"
[recoveryLeft1]: ./reportPics/center_2018_01_15_12_02_17_227.jpg "Example of Recovery from left right track"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode on both tracks

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py unmodified script for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network using all collected data
* video.mp4 Recording of autonomous run on left track
* video1.mp4 Recording of autonomous run on right track 
* writeup_report.md this document
* Record.zip, RecordOpposite.zip, Recovery.zip - Data collected for left track
* Record1.zip, Record1Opposite.zip, Recovery1.zip - Data collected for right track


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
model.h5 was achieved by training network for 25epochs 
```sh
python model.py --dataPath ./Record,./Recovery,./RecordOpposite,./Record1,./Recovery1,./Record1Opposite --epochs 25 --offset 0.15 --output model
```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An model based on [NVIDIA structure](https://arxiv.org/pdf/1704.07911.pdf) was used

My model consists of a convolution neural network with 5x5 filter sizes and depths 24, 36 and 48 (model.py lines 62-66) and two 3x3 filter with depth 64 (model.py lines 68-70) followed with 3 fully connected layers (model.py lines 72-76) 

The model includes RELU layers to introduce nonlinearity (model.py lines 62,54,66,68,70). Data is normalized in the model using a Keras lambda layer (model.py line 60) and cropped using Keras Cropping2D (model.py line 61). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers between each convolution layer in order to reduce overfitting (model.py lines 63,65,67,69).

Plot of training model (model.py) for 25 epochs:
![Training model for 25 epoch with dropout][25epDropout]

Plot of training model without dropout (trainNvidia.py) for 25 epochs:
![Training model for 25 epoch without dropout][25epNvid]

The model was trained and validated on different data sets to ensure that the model was not overfitting, see folders (Record.zip, RecordOpposite.zip, Recovery.zip, Record1.zip, Record1Opposite.zip, Recovery1.zip). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track, see video.mp4 and video1.mp4.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 79).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of centre lane driving in both directions, recovering from the left and right sides of the road on both tracks which resulted in 28455 training images.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create and train the model which can complete both tracks in autonomous mode

My first step was to use a convolution neural network model similar to the based on [NVIDIA structure](https://arxiv.org/pdf/1704.07911.pdf) I thought this model might be appropriate because it already demonstrated its usability in real word scenario. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set.(see trainNvidia.py) This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that after each convolution layer neurons are dropped out with 0.1 probability

The final step was to run the simulator to see how well the car was driving around left track which was successful but right track was much more tricky. There were a few sharp turns where the vehicle fell off the track to improve the driving behaviour I collected recovery data on both tracks. 

At the end of the process, the vehicle is able to drive autonomously around  both tracks without leaving the road.

#### 2. Final Model Architecture

My model consists of a convolution neural network with 5x5 filter sizes and depths 24, 36 and 48 (model.py lines 62-66) and two 3x3 filter with depth 64 (model.py lines 68-70). Between each layer there is dropout layer with 0.1 dropout probability to prevent overfitting.
Next 3 fully connected layers with sizes 100 50 and 1 are used (model.py lines 72-76) 
The model includes RELU layers to introduce nonlinearity (model.py lines 62,54,66,68,70). Data is normalized in the model using a Keras lambda layer (model.py line 60) and cropped using Keras Cropping2D (model.py line 61). 

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, I first recorded one lap on left track using centrer lane driving. Here is an example image of centrer lane driving:

![Centrer driving][centerDriving]

Next I recorded one lap of centrer driving on left track driving in opposite direction
![Centrer driving opposite][centerDrivingOpp]

I then recorded the vehicle recovering from the left side and right sides of the road back to centrer.

Recover from right:

![Recover from right][recoveryRight]

Recovery from left:

![Recovery from left][recoveryLeft]

Then I repeated this process on right track in order to get more data points.
Centrer driving:
![Centrer driving][centerDriving1]

Right track in opposite direction:

![Centrer driving opposite][centerDrivingOpp1]

Right track Recover from right:

![Recover from right][recoveryRight1]

Right track Recovery from left:

![Recovery from left][recoveryLeft1]


After the collection process, I had 28455 number of data points. I then converted images from BGR to RGB format.

Finally I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. After 10 epochs validation loss started dropping slower than training loss but it was still dropping at 25 epoch/ I used an adam optimizer so that manually training the learning rate wasn't necessary.
