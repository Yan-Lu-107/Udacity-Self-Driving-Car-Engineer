# README


[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)



**Advanced Lane Finding Project**

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

Check out the Video of the Advanced Lane Finding Result (click for full video): [![Advanced Lane Finding Result ](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/example/Advanced%20Lane%20Finding.gif)](https://youtu.be/H50zBnFf17c)


The Steps
The steps of this project are listed below. You can have a look at Advanced_Lane_Lines.ipynb for the code.

### Distortion Correction

The images for camera calibration are stored in the folder called camera_cal. I compute the camera matrix and distortion co-efficients to undistort the image.
![image1](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/calibration_output.jpg?raw=true)
![image2](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/undistion_output.jpg?raw=true)
### Gradients and color thresholds. I applied thresholds on gradients and colors (in RGB and HLS color spaces) to obtain a binary thresholded image.
![image3](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/color_thresh.jpg?raw=true)
###Perspective transform ("birds-eye view"). After manually examining a sample image, I extracted the vertices to perform a perspective transform. The polygon with these vertices is drawn on the image for visualization. Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
![image4](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/warped.jpg?raw=true)
### Detect lane pixels (sliding window search). I then perform a sliding window search, starting with the base likely positions of the 2 lanes, calculated from the histogram. I have used 10 windows of width 100 pixels.
The x & y coordinates of non zeros pixels are found, a polynomial is fit for these coordinates and the lane lines are drawn.
![image5](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/fit_polynomial.jpg?raw=true)
### Searching around previosly detected lane line Since consecutive frames are likely to have lane lines in roughly similar positions, we search around a margin of 50 pixels of the previously detected lane lines.
![image6](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/addWeighted.jpg?raw=true)
### Inverse transform and output For the final image we:
Paint the lane area
Perform an inverse perspective transform
Combine the precessed image with the original image.
![image7](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/addInfo.jpg?raw=true)
### Example Result We apply the pipeline to a test image. The original image and the processed image are shown side by side.



