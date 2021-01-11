# Udacity-Self-Driving-Car-Engineer

The goal is to write a software pipeline to identify the road lane boundaries in a video.

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

The Video
The pipeline is applied to a video. Click on the image to watch the video or click here. You will be redirected to YouTube.



[![Watch the video](https://github.com/Yan-Lu-107/Udacity-Self-Driving-Car-Engineer/blob/main/P2%20Advanced%20Lane%20Finding/Output_Process_Image/addWeighted.jpg?raw=true)](https://youtu.be/H50zBnFf17c)

