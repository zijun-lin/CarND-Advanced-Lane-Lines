## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[//]: # (Image References)

[image1]: ./output_images/0_camera_cal.jpg "Undistorted"
[image2]: ./output_images/1_distortion_correct.jpg "Road Transformed"
[image3]: ./output_images/2_create_binary_image.png "Binary Example"
[image4]: ./output_images/3_perspective_transform.jpg "Warp Example"
[image5]: ./output_images/4_identify_lane_lines.jpg "Fit Visual"
[image6]: ./output_images/5_final.jpg "Output"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the **Part 0** of P2_step.ipynb file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code of this step is containedin **Part 1** of P2_step.ipynb file.
Firest, I load the camera calibration and distortion coefficients and than using the `cv2.undistort()` function and obtained this result, as shown below. 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code of this step is containedin in **Part 2** of P2_step.ipynb file.
I used a combination of color and gradient thresholds to generate a binary image. Here's an example of my output for this step. We use different way to obtain these binary image, sunch as, Sobel on x and y direction, threshold to the overall magnitude and direction of the rradient, threshold in the HLS color space.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which is containedin in **Part 3** of P2_step.ipynb file.
The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 582, 460      | 320, 0        | 
| 205, 720      | 320, 720      |
| 1108, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code of this step is containedin in **Part 4** of P2_step.ipynb file.
First, I calculate the histogram of the bottom half of the image and finds the bottom-most x position of the left and right lane lines, than obtain the pixel which is believed to be the lane pixels by using sliding window. Finally, use the Numpy `polyfit()` method fits a second order polynomial to the set of pixels. The image below demonstrates how this process works:
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
The code of this step is containedin in **Part 5** of P2_step.ipynb file. The details of how to calculate the curvature of the lane and the position of the vehicle are described in it.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
I implemented this step in in **Part 5** of P2_step.ipynb file. Here is an example of my result which combine the above step:
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
For the video part, I defined a class called Line to save and process the lane line parameters. I set a buffer to process last n frame data. If the line is detected, the sliding window search is a little bit different from previous ways, just need to search a margin around the previous lane line position. I use `find_lane_pixels_from_image()` and `find_lane_pixels_search_around`. to find the line pixels.
I use `sanity_check()` function to check the pixels of lane line is correctly. If the pixels is good, save these parameters and update fit coefficients. If is bad, discard it. Finally, I use the Numpy `polyfit()` method fits a second order polynomial to the set of pixels.

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline will fail when the lighting conditions are bad and the lane line discoloration. To make the algorithm more robust, we can use the dynamic threshold in the process of creating the binary image, such as, the separate threshold parameters for different horizontal slices of the image.