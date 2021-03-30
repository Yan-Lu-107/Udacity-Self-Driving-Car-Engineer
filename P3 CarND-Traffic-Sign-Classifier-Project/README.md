## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/TrafficSign_Orignal.png "Traffic Sign Orignal"
[image2]: ./examples/distribution.png "Distribution"
[image3]: ./examples/ImagePre-process.png "Image pre-process"
[image4]: ./examples/NewImageFromWebsite.png "New Image from Website"
[image5]: ./examples/Prediction.png "Prediction"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P3%20CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here are some example of original traffic sign images.
![alt text][image1]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute in each label.
![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to improve performance.

Here is an example of a traffic sign image before and after grayscaling.
![alt text][image3]

As a last step, I normalized the image data by dividing 255 because neural networks work better if the input(feature) distributions have mean zero. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer					|     Description								| 
|:---------------------:|:---------------------------------------------:| 
| Input					| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling			| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x12	|
| RELU			 		|												|
| Max pooling			| 2x2 stride,  outputs 5x5x12					|
| Fatten				| To connect to fully-connected layers			|
| Fully-connected Layer1| outputs 150 									|
| Fully-connected Layer2| outputs 100 									|
| Fully-connected Layer3| outputs 43 									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a LeNet model and 100 epochs and set batch size at 128 and learning rate at 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of  0.950
* test set accuracy of 0.930


If an iterative approach was chosen:
At the beginning, I used a LeNet model without pre-process the image and set epochs at 30. I achieved a very low accuracy. I changed the size of kernal and the depth of convolution layer, but the result was not large. After adding the grayscaling and normalizing the image data, the result increased dramatically to 0.9. Then I set epochs to 100 and got a higher accuracy.

 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 German traffic signs that I found on the web:

![alt text][image4] 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					|		Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead 		 		 	 			| Turn right ahead 		 		 	 			|
| Right-of-way at the next intersection 		| Right-of-way at the next intersection 		|
| Road narrows on the right						| Dangerous curve to the right 					|
| Stop		 		 		 					| Stop		 		 		 					|
| No entry		 	 		 					| No entry		 	 		 					|
| Speed limit (60km/h) 		 		 			| Speed limit (80km/h) 		 		 			| 
| Go straight or right 							| Go straight or right 							|
| Double curve		 		 					| Beware of ice/snow 	 		 				|
| No passing for vehicles over 3.5 metric tons 	| No passing for vehicles over 3.5 metric tons 	|
| Traffic signals		 		 				| Traffic signals		 		 				| 
| Road work 	 								| Road work 	 								|
| General caution		 		 	 			| General caution		 		 	 			| 



The model was able to correctly guess 9 traffic signs, which gives an accuracy of 75%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead								| 
| 0.00     				| Go straight or left							|
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|
| 0.00				    | Speed limit (50km/h)							|

For the second image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way at the next intersection			| 
| 0.00     				| Pedestrians									|
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|
| 0.00				    | Speed limit (50km/h)							|

For the third image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Dangerous curve to the right					| 
| 0.00     				| Beware of ice/snow							|
| 0.00				    | Road narrows on the right						|
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|


For the fourth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999995         		| Stop											| 
| 0.000003     			| Speed limit (30km/h)							|
| 0.000002				| Speed limit (50km/h)							|
| 0.00					| Turn right ahead								|
| 0.00	      			| Yield			 								|


For the fifth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No entry					| 
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|
| 0.00				    | Speed limit (50km/h)							|
| 0.00	      			| Speed limit (60km/h)			 				|

For the sixth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (80km/h)							| 
| 0.00     				| Speed limit (60km/h)							|
| 0.00				    | End of speed limit (80km/h)					|
| 0.00					| Speed limit (50km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|


For the seventh image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Go straight or right							| 
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|
| 0.00				    | Speed limit (50km/h)							|
| 0.00	      			| Speed limit (60km/h)			 				|

For the eighth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Beware of ice/snow							| 
| 0.00     				| Children crossing								|
| 0.00				    | Right-of-way at the next intersection			|
| 0.00					| Bicycles crossing								|
| 0.00	      			| Road narrows on the right		 				|

For the ninth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No passing for vehicles over 3.5 metric tons	| 
| 0.00     				| Priority road									|
| 0.00				    | Speed limit (100km/h)							|
| 0.00					| End of no passing by vehicles over 3.5 metric tons|
| 0.00	      			| No passing			 						|


For the tenth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Traffic signals								| 
| 0.00     				| General caution								|
| 0.00				    | Go straight or left							|
| 0.00					| Right-of-way at the next intersection			|
| 0.00	      			| Road narrows on the right	 					|

For the eleventh image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Road work					| 
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|
| 0.00				    | Speed limit (50km/h)							|
| 0.00	      			| Speed limit (60km/h)			 				|

For the twelfth image:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| General caution								| 
| 0.00					| Speed limit (20km/h)							|
| 0.00	      			| Speed limit (30km/h)			 				|
| 0.00				    | Speed limit (50km/h)							|
| 0.00	      			| Speed limit (60km/h)			 				|

