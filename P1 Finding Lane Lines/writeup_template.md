# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


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


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to choose a suitable mask region that can fit all situation

Another potential improvement could be to increase the power of the ployfit.
