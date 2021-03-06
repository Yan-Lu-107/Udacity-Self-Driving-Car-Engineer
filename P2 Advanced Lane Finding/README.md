# Advanced Lane Finding Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./example/Advanced%20Lane%20Finding.gif "video gif"
[image2]: ./Output_Process_Image/calibration_output.jpg "video gif"
[image3]: ./Output_Process_Image/undistion_output.jpg "video gif"
[image4]: ./Output_Process_Image/color_thresh.jpg "video gif"
[image5]: ./Output_Process_Image/warped.jpg "video gif"
[image6]: ./Output_Process_Image/undistion_output.jpg "video gif"
[image7]: ./Output_Process_Image/fit_polynomial.jpg "video gif"
[image8]: ./Output_Process_Image/addWeighted.jpg "video gif"
[image9]: ./Output_Process_Image/addInfo.jpg "video gif"



The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Implement plausibilization/tracking mechanisms for continuous lane detection and processing based on the video
* Feed the project video to the developed pipeline and visualize the estimated lane as well as the estimated curvatures (left and right) and the vehicle's offset from the lane center


Check out the Video of the Advanced Lane Finding Result (click for full video): [![alt text][image1]](https://youtu.be/H50zBnFf17c)

### Distortion Correction

The images for camera calibration are stored in the folder called camera_cal. I compute the camera matrix and distortion co-efficients to undistort the image.

![alt text][image2]

![alt text][image3]

### Gradients and color thresholds. I applied thresholds on gradients and colors (in RGB and HLS color spaces) to obtain a binary thresholded image.
![alt text][image4]

### Perspective transform ("birds-eye view"). After manually examining a sample image, I extracted the vertices to perform a perspective transform. The polygon with these vertices is drawn on the image for visualization. Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
![alt text][image5]

### Detect lane pixels (sliding window search). I then perform a sliding window search, starting with the base likely positions of the 2 lanes, calculated from the histogram. I have used 10 windows of width 100 pixels.
The x & y coordinates of non zeros pixels are found, a polynomial is fit for these coordinates and the lane lines are drawn.
![alt text][image6]

### Searching around previosly detected lane line Since consecutive frames are likely to have lane lines in roughly similar positions, we search around a margin of 50 pixels of the previously detected lane lines.
![alt text][image7]

### Inverse transform and output For the final image we:
Paint the lane area
Perform an inverse perspective transform
Combine the precessed image with the original image.
![alt text][image8]

### Example Result We apply the pipeline to a test image. The original image and the processed image are shown side by side.



