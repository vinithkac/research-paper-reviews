<!-- SIFT -->

# Distinctive Image Features from Scale-Invariant Keypoints.

The is a review of the paper **Distinctive Image Features from Scale-Invariant Keypoints** by **David G. Lowe** accepted by *International Journal of Computer Vision, 2004*.

## Introduction
This paper mainly focus on extracting features of objects and try to match similar objects in different pictures based on particular interest points.
Images are transformed to large collection of local feature vectors which are **invariant** to:

* Illumination(Brightness)
* Scaling
* Rotation
* Viewing direction
* Image noise
* Occlusion 
* Clutter
  
Those features extracted are also  **distinctive or discriminative** to a particular object.
They are **efficient** enough to be implemented in real time.
And these features can be used in computer vision tasks such as :
* Image alignment and building panoramas
* 3D reconstruction
* Motion tracking
* Object recognition
and more..

To minimize the computational power, a cascade filtering approach in which only those features that pass through the previous layers are computer further.

## Detection stages:

1. Scale-space extrema detection
1. Keypoint localization
1. Orientation assignment
1. Generation of keypoint descriptor

SIFT features are first extracted from reference images and stored in a database and the features extracted from the new images is compared with features retrieved from the database to find the match by similarity.
Similarities between features are calculated by Fast nearest-neighbours algorithms discussed in this paper.

Even though the number of features are large, they are distinctive such that a single match can be found with a certain confidence.
Although an object is not approved until 3 or more local interest points matches with features for the same object from the previous database.
Before that, an affine approximation using least squares is made for the pose of the object and features vary much from the pose are considered as outliers and are rejected.
So that it can be confident with the recognized object.


## Scale-space extrema detection:
Using **Difference-of-Gaussian** function, it identifies keypoints(interest points) that are robust to orientation and scale.
Interest points are identified where there is a local extrema of *difference-of-Gaussian filters at different scales using a continuous function called as scale space.
This scale-space function is used to evaluate a keypoint qualitatively which is achieved by convoluting the input image with a Gaussian mask.
Gaussian mask is proven to get rid of unwanted details from a image at a given scale.
Here we generate progressive blurred images within a scale(octave) and reduce the image size(next octave) to half and do the same.(See Figure 1)

**Figure 1**

![Figure1](https://miro.medium.com/max/1280/0*ZlcI6l4Z6eNlfspE.jpg "Figure 1")



![eqn1](https://latex.codecogs.com/png.latex?L%28x%2Cy%2C%5Csigma%29%20%3D%20G%28x%2Cy%2C%5Csigma%29*I%28x%2Cy%29)

where,

![eqn2](https://latex.codecogs.com/png.latex?G%28x%2Cy%2C%5Csigma%29%20%3D%201/%282%5Cpi%20%5Csigma%20%5E2%29exp%5E%7B-%28x%5E2&plus;y%5E2%29/%5Csigma%20%5E2%7D)

is a variable scale Gaussian, the result of convolving an image with a difference-of-Gaussian filter.

![eqn3](https://latex.codecogs.com/png.latex?G%28x%2Cy%2Ck%5Csigma%29%20-%20G%28x%2Cy%2C%5Csigma%20%29)

*k => constant multiplicative factor*
 
 is given by,

![eqn4](https://latex.codecogs.com/png.latex?D%28x%2Cy%2C%5Csigma%20%29%20%3D%20L%28x%2Cy%2Ck%5Csigma%20%29-L%28x%2Cy%2C%5Csigma%20%29)

Or, simply difference of images with a Gaussian mask within a scale or octave for different sigma values(Gaussian parameterized by standard deviation).(See Figure 2)

**Figure 2**

![figure2](https://devanginiblog.files.wordpress.com/2016/05/sift1.png?w=840 "Figure2")

*The initial image is incrementally convoluted with Gaussian mask differ by a factor of 'k'.*
*A single octave sigma values range from ![eqn4.1](https://latex.codecogs.com/png.latex?%5Csigma%20%5Ctext%7B%20to%20%7D%202%5Csigma)*
*The image size is down-sampled by a factor of 2 for every octave which produces an image 1/4 of the previous one. (Each pixel in the new octave is mean of 4 pixels of the previous one)*.

The difference-of-Gaussian function gives similar results to the scale-normalized Laplacian of Gaussian.
![eqn5](https://latex.codecogs.com/png.latex?%5Csigma%20%5E2%5Ctriangledown%20%5E2G)

*Laplacian of Gaussian is a method often used in edge detection for images smoothened by a Gaussian mask*
![eqn6](https://latex.codecogs.com/png.latex?%5Ctriangledown%20%5E2G). (without scaling).
*It calculates second spatial derivative of Gaussian smoothened image which eliminates regions where there is no change in intensity (Gradient = 0)*.
 
It is found the, by normalizing using a factor of ![eqn6](https://latex.codecogs.com/png.latex?%5Csigma%20%5E2) LoG gives results which are Scale invariant and are better than Hessian or Harris corner functions.

![eqn7](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20%5Csigma%20%7D%20%3D%20%5Csigma%20%5Ctriangledown%20%5E2G)

So to relate it with D:

![eqn8](https://latex.codecogs.com/png.latex?%5Csigma%20%5Ctriangledown%20%5E2G%20%3D%20%5Cfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20%5Csigma%20%7D)

*Normalizing factor being sigma instead of* ![eqn6](https://latex.codecogs.com/png.latex?%5Csigma%20%5E2).

For difference of nearby scales,

![eqn9](https://latex.codecogs.com/png.latex?%5Csigma%20%5Ctriangledown%20%5E2G%20%3D%20%5Cfrac%7B%5Cpartial%20G%7D%7B%5Cpartial%20%5Csigma%20%7D%20%5Capprox%20%5Cfrac%7BG%28x%2Cy%2Ck%5Csigma%29%20-%20G%28x%2Cy%2C%5Csigma%20%29%7D%7Bk%5Csigma%20-%5Csigma%20%7D)

therefore,

![eqn10](https://latex.codecogs.com/png.latex?G%28x%2Cy%2Ck%5Csigma%29%20-%20G%28x%2Cy%2C%5Csigma%20%29%20%5Capprox%20%28k-1%29%5Csigma%20%5E2%5Ctriangledown%20%5E2G)

The factor (k-1) is constant over all scales.

#### Local extrema detection
To detect a local maxima or minima of DOG, each point is compared with its eight neighbours in the current image, nine in image above and below.(See Figure 2)
The point is considered as a keypoint only if the point is larger or smaller than all the other points (local maxima or minima) across scales.
It is found that no minimum spacing of samples extracts all the extrema from an image and quite unstable when it comes to features that are closer.
An optimum solution for frequency of sampling and image domains should be found that trades off efficiency with correctness.

**Figure 2**

*Sub-sampling with with Gaussian pre-filtering*


![Figure 2](https://img.favpng.com/9/0/24/scale-invariant-feature-transform-scale-invariance-algorithm-feature-detection-deep-learning-png-favpng-XrEJdAyqL9hsis4M2cgjwrtXH.jpg "Fiure2")

*Octave 1 uses scale sigma*
*Octave 2 uses scale 2 sigma*


#### Frequency of sampling in a scale domain
The sampling frequency that maximizes extrema stability is calculated experimentally by simulating a matching task on 32 real images.
Those images were transformed using rotation(random angle), scaling(between 0.2 and 0.9), shearing, illuminating and addition of noise(0.1% on every pixel).
It is found that, sampling 3 scales per octave yields the best results in terms of both keypoints which found matching location and scale and keypoints which found nearest descriptors in the database.
By increasing scales per octave, the number of keypoints detected goes up, but they were unstable and hard time finding matching keypoints on the location and in database.
Hence this paper chooses 3 samples per octave as the optimum value.

#### Frequency of sampling in the spatial domain
By plotting smoothing for octave against repeatability %, it is found that, higher the sigma value the number of matching descriptor found.
But taking computation into consideration, an optimal value for sigma = 1.6 is chosen in the paper.

It is found by experiment that, for the first octave, if the image is doubled in size blurred, the algorithm produces 4 time more valid keypoints.
But no further improvements notice while further increasing the size. 


## Keypoint localization
Now that we have much less points than the number of pixels of the whole image, this also include some bad keypoints.
There are three possibilities for a keypoint to pass through these steps.
* An error due to noise.
* An edge.
* A corner.

In this step errors due to noise and edge should be filtered out.
Only in a corner, gradients on both the sides are greater.
So if both the gradients are big enough, they are allowed to pass. Otherwise rejected.
The authors attempt to eliminate(filter) the keypoints by finding those who have low contrast comparatively and those in edges by calculating subpixel intensities.

Here they use Taylor series expansion(up to quadratic terms) of scale space to fit a 3D quadratic to a keypoint to interpolate the maxima or minima.
The equation is shifted to make sure that the origin is at the keypoint.

![eqn11](https://latex.codecogs.com/png.latex?D%28x%29%20%3D%20D%20&plus;%20%5Cfrac%7B%5Cpartial%20D%5ET%7D%7B%5Cpartial%20x%5E2%7D&plus;%5Cfrac%7B1%7D%7B2%7DT%5Cfrac%7B%5Cpartial%5E2D%7D%7B%5Cpartial%20x%5E2%7Dx)

Offset => ![eqn12](https://latex.codecogs.com/png.latex?x%20%3D%20%28x%2Cy%2C%5Csigma%20%29%5ET)

The location of the keypoint is calculated by taking the derivative of this function.

![eqn12](https://latex.codecogs.com/png.latex?%5Cwidehat%7Bx%7D%20%3D%20-%5Cfrac%7B%5Cpartial%5E2D%5E%7B-1%7D%7D%7B%5Cpartial%20x%5E2%7D%5Cfrac%7B%5Cpartial%20D%7D%7B%5Cpartial%20x%7D)


Sample points which are closer are interpolated to determine its position accurately.
To reject a keypoint for unstable extrema, we use the function:

![eqn13](https://latex.codecogs.com/png.latex?D%28%5Cwidehat%7Bx%7D%29%29%20%3D%20D%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B%5Cpartial%20D%5ET%7D%7B%5Cpartial%20x%7D%5Cwidehat%7Bx%7D)

This is how they refine keypoints by using the intensities at sub-pixel locations. 
For a keypoint to be accepted, this value should have a magnitude of less than 0.03.

#### Eliminating edge responses
After rejecting keypoints with low contrast, points that are poorly localized along the edges should also be removed.
Those points can be identified by Hessian matrix(H), computed at location and scale of the keypoint.

![eqn14](https://latex.codecogs.com/png.latex?H%20%3D%20%5Cbegin%7Bpmatrix%7DD_%7Bxx%7D%20%26%20D_%7Bxy%7D%5C%5C%5C%20D_%7Bxy%7D%20%26%20D_%7Byy%7D%5Cend%7Bpmatrix%7D)

Since we are only concerned about the ratio of the eigenvalues, they are not computed explicitly, instead

*Sum of the values => Trace of the matrix*

*Product of the values => Determinant of the matrix*

![eqn15](https://latex.codecogs.com/png.latex?Tr%28H%29%20%3D%20D_%7Bxx%7D%20&plus;%20D_%7Byy%7D%20%3D%20%5Calpha%20&plus;%5Cbeta)

![eqn16](https://latex.codecogs.com/png.latex?Det%28H%29%20%3D%20D_%7Bxx%7DD_%7Byy%7D-%7BD_%7Bxy%7D%7D%5E2%3D%5Calpha%20%5Cbeta)

The point is discarded if there is a negative determinant value as the curvature has different signs.
r be the ratio between largest and smallest eigenvalue. then,

![eqn17](https://latex.codecogs.com/png.latex?%5Cfrac%7BTr%28H%29%5E2%7D%7BDet%28H%29%7D%3D%5Cfrac%7B%28%5Calpha%20&plus;%5Cbeta%29%5E2%20%7D%7B%5Calpha%20%5Cbeta%20%7D%3D%5Cfrac%7B%28r%5Cbeta%20&plus;%5Cbeta%29%5E2%20%7D%7Br%5Cbeta%20%5E2%7D%3D%5Cfrac%7B%28r&plus;1%29%5E2%7D%7Br%7D)

This ratio is minimum when the two eigenvalues are the same. To check whether the ratio is below a threshold(r = 10 in this paper), we only need to check,

![eqn18](https://latex.codecogs.com/png.latex?%5Cfrac%7BTr%28H%29%5E2%7D%7BDet%28H%29%7D%20%3C%20%5Cfrac%7B%28r&plus;1%29%5E2%7D%7Br%7D)

This further eliminates keypoints based on this equation.

## Orientation assignment
To achieve invariance to image rotation, orientations are assigned to each keypoints based on local image properties.
A gradient orientation histogram is computed around each of the keypoint(neighbours).
As we know the scale(window size) of the keypoint detected, it is used to select the Gaussian smoothed image(L) with the closest scale so that all the computations are computed in a scale invariant manner(pixels contributing to the computation corresponding to a keypoint).
The contribution of each neighbouring pixel is weighted by the gradient magnitude and a Gaussian window with a Sigma that is 1.5 times the scale of the keypoint.

 
For each L(x,y), magnitude m(x,y) and orientation is calculated using pixel differences.
 
magnitude, ![eqn19](https://latex.codecogs.com/png.latex?m%28x%2Cy%29%3D%5Csqrt%7B%28L%28x&plus;1%2Cy%29-L%28x-1%2Cy%29%29%5E2&plus;%28L%28x%2Cy&plus;1%29-L%28x%2Cy-1%29%29%5E2%7D)

orientation, ![eqn20](https://latex.codecogs.com/png.latex?%5Ctheta%20%28x%2Cy%29%20%3D%20%5Ctan%5E%7B-1%7D%28%28L%28x%2Cy&plus;1%29-L%28x%2Cy-1%29%29/L%28x&plus;1%2Cy%29-L%28x-1%2Cy%29%29)

The orientations has 36 bins(each 10 degrees) covering all the directions(360 degrees) around a point.
Highest Peak in the histograms correspond to the dominant orientation, any other peak within 80% of the highest peak is also considered while creating the orientation.
Finally a parabola is fit to the top 3(within 80%) histogram values to interpolate the position accurately.
All the properties of the keypoint are measured relative to the keypoint orientation.This ensure rotation invariance.
By experiments, authors concluded that it has resistance to noise to some extent(10%).

## Local image descriptor
Now that we have keypoints that are scale, rotation invariant and has a corresponding orientation to it, now we can compute a descriptor that are highly distinctive and invariant to illumination and 3D viewpoints.
#### Descriptor mechanism
Image gradient magnitudes and orientations around a keypoint location is taken into consideration using the same scale and the Gaussian blur level where the keypoint is found.
A Gaussian weighting function with sigma equal to one half the width of the descriptor window is used to assign a weight to magnitude of each sample point.
The samples is the window is summarized over 4*4 descriptor array which is computed from an 16*16 set of samples.
To give importance to samples nearer to the keypoint, an linear interpolation is used.
That is by multiplying each sample with a weight of *d-1* for each dimension, where d is the distance of the sample from the center.

To make the vector invariant to illumination changes, we first normalize it to unit length.
If a change occurs in image contrast, each pixel value is multiplies by a constant which will be handled while normalization.
Hence it becomes invariant to illumination changes since the gradient values are computed from difference of image pixels.

#### Descriptor testing 
To set the complexity of the descriptor, an optimize value for number of orientations(r) and size of the window(n) need to be found.
The size of the descriptor vector is r*n*n. 
It is found that as the complexity increases, the more the distinctive it is, but it describes the object so well that it becomes invariant to image shearing, noise and occlusion.
Vector matching from the database when n = 1 is 0.
It improves rapidly till the window size of n = 4 before it flattens.
And at n=4 nearly all given histogram orientations perform similar. Hence taking computation into consideration, r=4 is selected.
Hence the dimension of the descriptor vector is 4*4*8 = 128 dimensions.   

#### Sensitivity to affine changes
The features are not yet affine invariant.
By experiments, the final matches found for the keypoints are just little over 50% when there is a 50 degree change in viewpoint.
One way to do that is by having a bigger database with transformed versions 0f the training images to match up to 60 degree viewpoint changes.
By this way no extra computation on the feature vector is required.
But this increases the database 3 times of the original one.

#### Matching to large databases
In this paper, the keypoints about 40,000 are extracted from 32 images.
It is found that, matching is not perfect as the change in viewpoint angle is higher.
By experiments, it is found that features are affine invariant up to rotation of 30 degrees after which the keypoints matching in the database decreases significantly.


## Application to object recognition
By using this features, objection recognition can be done is a set of steps.
First, Keypoints are matched with the database.
The independent keypoints are clustered so that at least 3 features could agree on an object).
Also the pose and geometric fit is checked before it is declared as an object.

#### Keypoint matching algorithm
Key points are matched by a different kind of nearest neighbors algorithm.
The nearest neighbor is the keypoint vector which has the minimum Euclidean distance.
Here a keypoint is matched by comparing the ratio of distance of the closest neighbour to the second closest neighbor.
By doing this, we can also compute the  number of false matches between the range of 1st and 2nd neighbours which helps in reducing false matches.
The ratio of 0.8 is selected. Any two neighbours within ratio of 0.8 is selected as a match.
This eliminates 90% of incorrect matches.

#### Efficiency in matching nearest neighbors
To increase the efficiency in computing the nearest neighbors, an algorithm called Best-Bin-First is used.
A heap-based priority queue is conducted on the feature space starting from the query location.
This gives an approximate matching value with a low cost but results in 5% loss in correct matches.

#### Clustering with Hough transform
On an average, an image has about 20,000 features.
After finding nearest neighbors, we can eliminate a lot of features from them.
Now that we have features that match, we can use them to classify them into objects.
The localized region from the ratio of neighbors accounts for 99% of false matches.
To match the features to a particular object, the pose of the object is matched with the features aligned in the feature space.
Here they used Hough transform which performs well better than Least squares or RANSAC.
**The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure.**
Each of the keypoints has these parameters : 2D location, scale, orientation and the keypoint matched with it.
They create a Hough transform to predict orientation and scale from the match hypothesis.

This basically means, identifying clusters with at least 3 matching points in a bin.
And then, least squares is performed to verify the geometry between points of training image and new image.
An other approach is taken here instead which is the Fundamental matrix.
The affine transformation of a model point [x y] the transpose to an image point [u v] to the transpose can be written as:

![eqn21](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20u%5C%5C%20v%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20m1%20%26%20m2%5C%5C%20m3%20%26%20m4%20%5Cend%7Bbmatrix%7D%5Cbegin%7Bbmatrix%7D%20x%5C%5C%20y%20%5Cend%7Bbmatrix%7D%20&plus;%20%5Cbegin%7Bbmatrix%7D%20t_x%5C%5C%20t_y%20%5Cend%7Bbmatrix%7D)

the affine rotation, scale and stretch are represented by parameters m.

to find the transformation of the point, this can be written as,

![eqn22](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20x%20%26%20y%20%26%200%20%26%200%20%26%201%20%26%200%20%5C%5C%200%20%26%200%20%26%20x%20%26%20y%20%26%200%20%26%201%5C%5C%20%26%20.%20%26%20.%20%26%20.%20%26%20%26%20%5C%5C%20%26%20.%20%26%20.%20%26%20.%20%26%20%26%20%5Cend%7Bbmatrix%7D%20%5Cbegin%7Bbmatrix%7D%20m_1%5C%5C%20m_2%5C%5C%20m_3%5C%5C%20m_4%5C%5C%20t_x%5C%5C%20t_y%20%5Cend%7Bbmatrix%7D%3D%20%5Cbegin%7Bbmatrix%7D%20u%5C%5C%20v%5C%5C%20.%5C%5C%20.%5C%5C%20.%5C%5C%20%5Cend%7Bbmatrix%7D)

the dots represent other matches. As you can see there are 6 translation parameters indicating at least 3 matched points are required to provide a solution.

Consider this as a system of linear equations, Ax = b. Then, least squares solutions for x is given by,

![eqn23](https://latex.codecogs.com/gif.latex?x%20%3D%20%5BA%5ETA%5D%5E%7B-1%7DA%5ETb).

Now that we have least squares error, we compare them wih hough transform bins. And if there are more than 3 points remaining, its is accepted as the detected object.
The final decision is based on Bayesian analysis given the projected size, the number of features within the region and the accuracy of the fit to the pose.
A model is accepted only if the probability is greater than 0.98.


## Inferences
The method in this paper is not fully affine invariant.
The feature that are close together are unstable here.





 
