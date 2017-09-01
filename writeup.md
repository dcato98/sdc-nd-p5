# Udacity Self-Driving Car Nanodegree
---

## Project #5: Vehicle Detection
---

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Append binned spatial features, as well as histograms of color, to a HOG feature vector
* Normalize features and randomize a selection for training and testing
* Implement a sliding-window search and use a trained classifier to search for vehicles in images
* Create a heat map of recurring detections frame-by-frame to reject outliers and follow detected vehicles in a video stream
* Estimate a bounding box for vehicles detected

[//]: # (Image References)
[image1]: ./example_images/output_10_0.png
[image2]: ./example_images/output_10_1.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features` in lines 16-33 of [project_utils.py](/scripts/project_utils.py), which uses the `hog` function in the `skimage.feature` package.

All vehicle and non-vehicle training images are passed through this function to compute the HOG features in cell 6 of [Vehicle-Detection.ipynb](/scripts/Vehicle-Detection.ipynb). HOG features are later calculated on each test video frames at a variety of image scales.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled on the values in cell 5 of [Vehicle-Detection.ipynb](/scripts/Vehicle-Detection.ipynb) since they consistently produced a model with > 98% validation accuracy. I had originally settled on `pix_per_cell`=32, however, to improve computation time, I reduced this to 16 without a noticeable loss in accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in cell 7 of [Vehicle-Detection.ipynb](/scripts/Vehicle-Detection.ipynb). The parameter value `C`=0.05 was determined by a grid search using 3-fold cross validation.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After (probably far too much) experimentation with a variety of scales and x/y bounds, I settled on keeping it simple, using just 2 scales (1 and 1.5), both covering the full x-range and the limited y-range where cars can be found. For overlap, the original value of `cells_per_step`=2 didn't have enough overlap, so I set the `cells_per_step`=1 (line 225) in the `find_objects` function in [project_utils.py](/scripts/project_utils.py) which provided good overlap without a huge hit to performance.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales (1 and 1.5) using YUV color HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Performance was optimized by using HOG sub-sampling, a heat decay rate (instead of frame averaging), and reducing the number of windows to be searched. Here are some example images:

![example car detection 1][image1]
![example car detection 2][image2]
---

### Video Implementation

#### 1. Provide a link to your final video output.

Here's a [link to my video result](./project_video_with_cars.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In `process_frame`, lines 32-65 of [vehicle_tracker.py](/scripts/vehicle_tracker.py), I process the video image. After detecting sections of the frame where cars are likely (line 36), I filter out predictions with a confidence < 0.5 (line 42), then append the remaining sections together to form a single-frame heatmap (line 46) effectively combining overlapping bounding boxes. From here, predictions are clipped to 2.5 in order to limit the effect of any individual frame on predicting a car (line 48). Next, the running heatmap is downweighted by a constant factor, the frame heatmap is added to the recurring heatmap, and a Gaussian blur is applied to smooth the prediction (lines 50-52). Finally cars are identified by selecting the sections of the heatmap which exceed the threshold (lines 55-58). Setting the parameters, `heat_decay`=0.9 and `heat_thresh`=10 (in cell 5 of [Vehicle-Detection.ipynb](/scripts/Vehicle-Detection.ipynb)), provided a nice balance between keeping tight boundaries around the car and minimizing false positive car detections.

---

### Discussion

#### 1. Briefly discuss any issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is a little slow to identify cars at the edges of the image. Two techniques could improve this, 1) add extra pixels to the edges so that the pipeline makes more predictions at the edges, and/or 2) expand the field of view, either with additional cameras or a wider lens (probably the first), so that cars on the edge of view are seamlessly detected.

The pipeline is also slow to make predictions. As is, it would not make predictions fast enough for a car driving in real-time. Computing a bunch of expensive, hand-crafted features is, on the whole, not very robust. To make predictions on multiple frames per second, I think a more robust approach would be to use an end-to-end neural network which takes the raw image and the prior prediction as input and outputs the car locations. This is likely to be able to produce both a faster and more robust solution.
