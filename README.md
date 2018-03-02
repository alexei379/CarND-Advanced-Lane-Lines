**Advanced Lane Finding Project**

The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms & gradients to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

While implementing this project I was using course materials, Q&A session video and discussion forums provided by Udacity.

[//]: # (Image References)

[original_chessboard]: ./camera_cal/calibration1.jpg "Original chessboard"
[undistorted_chessboard]: ./output_images/calibration1.jpg "Undistorted chessboard"
[original]: ./test_images/test6.jpg "Original road"
[undistorted_road]: ./output_images/test6.jpg_0.jpg "Undistorted road"
[binary]: ./output_images/test6.jpg_1.jpg "Undistorted road"


[straight_unwarped]: ./output_images/straight_lines1.jpg_2_1.jpg "Straight lines unwarped"
[straight_warped]: ./output_images/straight_lines1.jpg_2_2.jpg "Straight lines warped"

[fit_visual]: ./output_images/test6.jpg_3.jpg "Fit Visual"

[plotted_warped]: ./output_images/test6.jpg_4.jpg "Plotted lane warped"
[plotted_unwarped]: ./output_images/test6.jpg_5.jpg "Plotted lane unwarped"
[result]: ./output_images/test6.jpg_7.jpg "Result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how they are addressed
You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion-corrected calibration image.

The code for this step is contained in IPython notebook located in "./camera_calibration.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied the resulting camera matrix `mtx` and distortion coefficients `dist` into `calibration_pickle.p` file.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

| Original Chessboard | Undistorted Chessboard |
|:-------------------:|:---------------------:| 
| ![original_chessboard] | ![undistorted_chessboard] |

### Pipeline (single images)

To test the pipeline I implemented `process_images` function in the `advanced_lane_pipeline.py` file. It reads test images from `test_images` directory and saves resulting images from each step.

#### 1. Provide an example of a distortion-corrected image.

I implemented it in `ImageTransformer.py` file in `undistort` method, which is a wrapper around `cv2.undistort`. The pipeline loads camera matrix `mtx` and distortion coefficients `dist` from `calibration_pickle.p` file, instantiates ImageTransformer and calls undistort to perform distortion correction. 

| Original Road | Undistorted Road |
|:-------------------:|:---------------------:| 
| ![original] | ![undistorted_road] |

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color, Sobel x and y operator thresholds to generate a binary image. Implementation is in `Thresholder.py` file, method `pipeline` contains calls to the appropriate functions with threshold parameters and combines the output into a resulting image.

![binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is implemented in method `warp()` of `ImageTransformer` class, that is implemented in `ImageTransformer.py`. The `ImageTransformer` gets configuration object in the constructor parameter. Configuration is defined in `Config.py`, it includes `warp_src` and `warp_dst` matrices. They are used to compute `M` matrix to perform the transformation.


```python
    img_shape = (720, 1280)

    warp_src = np.float32(
        [[(img_shape[1] / 2) - 470, img_shape[0] - 60],
         [(img_shape[1] / 2) - 72, img_shape[0] - 265],
         [(img_shape[1] / 2) + 72, img_shape[0] - 265],
         [(img_shape[1] / 2) + 470, img_shape[0] - 60]])

    warp_dst = np.float32(
        [[(img_shape[1] / 2) - 400, img_shape[0]],
         [(img_shape[1] / 2) - 400, 0],
         [(img_shape[1] / 2) + 400, 0],
         [(img_shape[1] / 2) + 400, img_shape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 170, 660      | 240, 720        | 
| 568, 455      |240, 0      |
| 712, 455     | 1040, 0      |
| 1110, 660      | 1040, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image (I used a different test image with straight lane markings to demonstrate this effect).

| Source        | Warped   | 
|:-------------:|:-------------:| 
| ![straight_unwarped] | ![straight_warped] | 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I implemented it in `Line.py` and `SlidingLaneFinder.py` files. In the pipeline, I create objects of `Line` class for left and right line and keep track of the last 5 measurements, including lane starting points, fitted x and y pixels and radius. The pipeline takes the thresholded warped image and sends it as input to `SlidingLaneFinder.find` method.
`SlidingLaneFinder.find` find finds peaks in the histogram to the left and right of the image center and uses it as the starting point of possible lane markings (averaged over last 5 frames). After this I find 20 "windows" for each lane moving from the starting point towards the possible next location (window width = 100px, minimum number of pixels to be considered as part of the line - 30 px). While moving the window, I keep track of non-zero pixels. After this detected points are added to the line object and based on last 5 frames I fit a second level polynomial through the points using numpy's `polyfit` function. If the resulting quadratic coefficient of left and right lane markings are within 0.0003 of each other, I consider the detected lane as a good one and keep found points, otherwise - discard.

`find` method accepts `debug` parameter, that allows rendering the results of lane markings detection. The output of it:
![fit_visual]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to the center.

I did this in `ResultVisualizer.py` file. I found the highway from the video on [Google maps](https://www.google.com/maps/@37.4451569,-122.2627474,1504m/data=!3m1!1e3), converted the length of lane marks and width of lanes in on satellite images from pixels to meters based on the scale ruler, and used the resulting measurements in meters to get conversion coefficients for the warped images.

Method `stamp_radius` converts detected points using the above coefficients, fits a new polynomial and calculates radius at lines 28/29. The radius is averaged over last 5 frames and between left and right lanes.

Method `stamp_offset` calculates offset by finding a difference between the center of the image and midpoint between the lane marks, converting them to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Plotting the result back to the original image is done in several steps: `draw_lane` in `ResultVisualizer.py` plots polygon bounded by the fitted lane marking points to warped space, method `unwarp` in `ImageTransformer.py` un-warps the lane back to original image space and `process_images` in `advanced_lane_pipeline.py` adds them together. As the final pass, `stamp_radius` and `stamp_offset` are invoked to add curvature radius and vehicle offset.

Here are the steps applied to the test image:

| Warped        | Un-warped   | 
|:-------------:|:-------------:| 
| ![plotted_warped] | ![plotted_unwarped] |

![result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The pipeline for video is implemented in `advanced_lane_pipeline.py` `process_video` function. It resembles the test image pipeline.
Here's a [link to my video result (YouTube)](https://www.youtube.com/watch?v=drMYbWS46yw)


---

### Discussion

#### 1. Briefly discuss any problems/issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The thresholds I found for using Sobel x/y operator combined with color thresholding work nicely on the project video. Hoewer additional technics like thresholding magnitude and direction of the gradient might be useful for more complex scenarios. Also, I am using sliding window in every frame, but that can be optimized using the information found from the previous frames. I average the results over the last several frames, which on one hand results in smoother output and keeps outliers from drastically affecting the picture, but might not work as well for situations with sudden movement like changing lanes.
