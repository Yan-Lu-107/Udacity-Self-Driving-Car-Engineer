# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />


[![Udacity - Self-Driving Car NanoDegree](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P1%20Finding%20Lane%20Lines/Finding%20Lane%20Lines.gif)](https://www.youtube.com/watch?v=H50zBnFf17c)


Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm. In this project lane lines in images will be detected using Python and OpenCV.   


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1. I converted the images to grayscale
2. Using gaussian function to smooth the edge
3. Detecting the edges with canny function
4. Applying mask to the region that covers two lanes
5. Draw the lines with setting the values of length, gaps, resolution
6. In order to draw a single line on the left and right lanes, I added the functions of lines that fits the lane curve
7. visualize the plot

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lane with a large curve.

Another shortcoming could be that if the lanes are not in the region that masks applied, it will go wrong.

