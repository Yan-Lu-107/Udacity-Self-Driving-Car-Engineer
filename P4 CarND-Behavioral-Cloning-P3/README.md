# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/TestImage.jpg "Original Image"
[image2]: ./examples/recoveryfromleft2center.gif "Recovery from Left to Center"
[image3]: ./examples/CroppedImage.jpg "Cropped Image"
[image4]: ./examples/FlippedImage.jpg "Flipped Image"
[image5]: ./examples/NvidiaModel.jpg "Model Visualization"
[image6]: ./examples/Loss1.png "Model Loss"
[image7]: ./examples/Loss2.png "Model Loss"
[image8]: ./examples/Loss3.png "Model Loss"
[image9]: ./examples/Loss4.png "Model Loss"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.(model.py lines 17-23) 

I collected the following data,
1. Several tracks of driving forwards and several rounds of driving counter-clockwise at the center of the lane
	To capture good driving behavior, I recorded two laps forwards along the road and another two laps counter-clockwise using center lane driving. Here is an example image of center lane driving:

	![alt text][image1]

2. Collecting the data of recovering from the left and right sides of the road
	I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to driving back to center when it is on the edge of the road. These images show what a recovery looks like starting from left side to the center:
	
![alt text][image2]

3. Flipping horizontally the images to augment the data and adjusting the steering angle
	For example, here is an image that has then been flipped:

	![alt text][image1]
	![alt text][image4]

4. Cropping useless background of the images

	![alt text][image1]
	![alt text][image3]

5. Taking use of left and right camera and setting the offset value to steering angle(the offset value is set to 0.3)

After the collection process, I had 18192 number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the lostest loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.



#### 2. An appropriate model architecture has been employed

The model used NVIDIA consists of five convolution layers with 5x5 and 3x3 kernel sizes and then follows four fully connected layers.(model.py lines 111-138) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 111). 

Layer 1: Conv layer with 24 5x5 filters, followed by ELU activation
Layer 2: Conv layer with 36 5x5 filters, ELU activation
Layer 3: Conv layer with 48 5x5 filters, ELU activation
Layer 4: Conv layer with 64 5x5 filters, ELU activation
Layer 5: Conv layer with 64 5x5 filters, ELU activation

Layer 6: Fully connected layer with 2112 neurons, ELU activation
Layer 7: Fully connected layer with 100 neurons, ELU activation
Layer 8: Fully connected layer with 50 neurons, ELU activation
Layer 9: Fully connected layer with 10 neurons, ELU activation
Layer 10: Fully connected layer with 1 neurons, ELU activation
	
![alt text][image5]

#### 3. Attempts to reduce overfitting in the model

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the loss for training set and validation set become so different with epoch increasing.The mean squared error on the training set becomes lower, but the mean squared error on the validation set are stable after epoch at 8. 
	
![alt text][image6]

After I changed epoch to 8, the validation loss increased after epoch 3. so I tried to add dropout.

![alt text][image7]

After dropout added, the loss chart looks much better, but the performance of the car is still not good.

![alt text][image8]

Then I tried to decrease the epoch and it turns out the car performance best at epoch 7.

![alt text][image9]

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 141).


#### 5. Final Test on the Simulator

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I recorded more about the part of the laps that recoverying from the side of the track. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.





