<!-- Haar cascade -->
# Haar cascade
#### Survey on *Rapid Object Detection using a Boosted Cascade of Simple Features*

## Introduction
This is a review of the paper *Rapid Object Detection using a Boosted Cascade of Simple
Features* by **Paul Viola, Michael Jones**.
This is the first object detection framework proposed that detects objects in real time (*15 frames per second*).
This is a machine learning approach where an actual classifier (cascade function) is trained using positive and negative images. 

## Problem statement
This paper works on a particular object - **Frontal Face**. But now there are fully trained models for other objects as well.
This paper being proposed long before the modern deep learning approaches,
Haar Cascades are still the most used face detection algorithm as they are faster compared to YOLO, SSD, HOG, etc...

## List of summarized papers
Rapid Object Detection using a Boosted Cascade of Simple Features by **Paul Viola, Michael Jones** published on  CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION 2001
| [paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf "Link to the paper") |
## Datasets
#### Training : 
Set of face and non-faces. 
Face training set of 4916 images scaled and aligned to 24*24 pixels. Face training set is doubles by mirroring faces which gives a new set of records to train.
The non-face sub-windows were extracted from 9544 images and also scaled to 24*24 pixels.

#### Validation : 
Evaluated on MIT+CMU test set.

  
## Paper summary
### Approach
This paper proposes a step by step approach to the final cascade classifier.
1. Representing the image in a new method -Integral Image
1. The learning algorithm - AdaBoost.
1. Cascading classifiers.

**A simple flow chart of the algorithm.**

![Flow chart](https://image.slidesharecdn.com/a5773c5e-f620-47e1-93ea-5d9f53714640-160525121315/95/face-detection-9-638.jpg?cb=1480060921 "Figure 1")

Input image shape : (384, 288), detected at 15 frames per second  on a conventional 700 MHz
Intel Pentium III.

#### Haar-Like Features
An object detection model detects an object in the picture by checking whether the features of the object is present.
Features are nothing but numerical information extracted from the images that can be used to distinguish one image from another.
For example, a histogram (distribution of intensity values) is one of the features that can be used (HOG) to define several characteristics of an image such as dark or bright image, the intensity range of the image, contrast, etc...
Haar features find some patterns(pixel intensities) that are common in objects(Face).(See Figure 2)
These features are similar to what we use now as convolutional kernels but resultant value calculation is different.
The value at any given x, y is calculated by sum(pixels in black) - sum(pixels in black).

![eqn](https://latex.codecogs.com/svg.latex?f%28x%2Cy%29%20%3D%20%5Csum_i%20%7Bp_b%28i%29%7D%20-%20%5Csum_i%20%7Bp_w%28i%29%7D)



**Figure 2**

![Haar features](https://miro.medium.com/max/2808/1*i8KJBkCHHgw7EuLNXS8--Q.png "Figure 2")

#### Integral Image
Integral image is a new way of representing an image.
Although haar-like rectangle features are less flexible and vulnerable to curves and edges of an image than *steerable features*, combined with integral image, they compensate through its speed of evaluation.
This allows fast feature evaluation at different scales very quickly.
Instead of working with image intensities, a sub window of the image is represented as integral image as shown in Figure 3.
Different scales is in the sense, Instead of scaling the image to different sizes like pyramids, we scale the features.
The base resolution of the detector is 24 * 24 and the set of rectangle features forms a basis over 180,000.
Considering the complete basis here is just 576(total pixels), 180,000 features for a sub-window is over-complete many times.
*Over-complete is term referred when a basis is still complete even after removal of a vector from the basis.*
Integral image at any location x, y is sum of pixels above and left of x, y inclusive.

**i(x,y)**  --> original image.

**ii(x,y)** --> integral image

![eqn](https://latex.codecogs.com/svg.latex?ii%28x%2Cy%29%20%3D%20%5Csum_%7B%7Bx%7D%27%5Cleq%20x%2C%7By%7D%27%5Cleq%20y%7D%20i%28%7Bx%7D%27%2C%7By%7D%27%29)

![eqn](https://latex.codecogs.com/svg.latex?s%28x%2C%20y%29%20%3D%20s%28x%2C%20y-1%29%20&plus;%20i%28x%2Cy%29)

![eqn](https://latex.codecogs.com/svg.latex?ii%28x%2Cy%29%3Dii%28x-1%2Cy%29&plus;s%28x%2Cy%29)

**s(x, y)**  --> cumulative sum row. And also s(x, -1) = 0, ii(-1, y) = 0.

For example: 

matrix : 

![eqn](https://latex.codecogs.com/svg.latex?%5Cbegin%7Bbmatrix%7D%201%20%26%205%5C%5C%202%20%26%204%20%5Cend%7Bbmatrix%7D) 

hence,

* at ii(-1, -1) = ii(-1, 0) = ii(0, -1) = ii(-1, 1) = ii(1, -1) = 0

* at (0, 0) :

        s(0, 0) = 0 + 1 => 1
        ii(0, 0) = 0 + 1 => 1
    
* at (0, 1):

        s(0, 1) = 1 + 5 => 6
        ii(0, 1) = 0 + 6 => 6
        
* at (1, 0) :
        
        s(1, 0) = 0 + 2 => 2
        ii(1, 0) = 1 + 2 => 3
     
* at (1, 1):

        s(1, 1) = 3 + 4 => 7
        ii(1, 1) = 5 + 7 => 12    


**Figure 3**

![integral image](https://it.mathworks.com/help/images/integral_image_a.png "Figure 3")



#### AdaBoost
AdaBoost algorithm is used here to construct a classifier using important features as it learns by boosting weak learners and gives importance(weights) to misclassified records adaptively.
An AdaBoost algorithm is constructed on a single stump(feature) for the whole features known as base learners and improvise on it.
The feature at the stump is selected by a criterion.
Here it is used for both feature selection and for training the classifier.
AdaBoost has proven generalization property and it does well when the records are in wide range.



#### Cascading classifiers
Cascading the classifiers is used to discard or exclude regions that are unlikely(low false negatives) to be a face and focus on regions where a frontal face is most likely to be present with a statistical guarantee.
This reduces the computational power wasted on the regions where the objects are unlikely to be present.



#### Learning
Image is passed through a classifiers in a sequence.
Only the sub-windows that are accepted by a classifier is passed on to the next one. 
They were able to reduce up to half of the regions to be evaluated by the final detector by using just 2 Haar-like features.
Three Kinds of features are used here:
1.Two-rectangle feature(s) (horizontal & vertical)
1.Three-rectangle feature (center)
1.Four-rectangle feature (diagonal)
(See Figure 6)

As we know, there are over 180,000 features for each sub-window, computing the complete set is expensive.
By experiments the authors found only a few feature can be combined to make a good classifier.
To find those features, a weak learner(stump) picks a single feature which separates positive and negative samples effectively.
*A stump or a base learner is a single node decision tree. The feature at the node is selected creating multiple stumps and the best is selected(minimum records are misclassified).*
Therefore a weak classifier, consists of a feature, a threshold and parity indicating the direction of inequality.

![eqn](https://latex.codecogs.com/svg.latex?h_j%28x%29%20%3D%20%5Cbegin%7Bcases%7D%201%26%5Ctext%7Bif%20%7Dp_jf_j%28x%29%20%3C%20p_j%5Ctheta_j%5C%5C%200%26%5Ctext%7Botherwise%7D%20%5Cend%7Bcases%7D)

For every 24*24 sub-window, a base learner is created.

* Given, examples of images ![eqn](https://latex.codecogs.com/svg.latex?%28x_1%2C%20y_1%29%2C...%2C%28x_n%2C%20y_n%29) where ![eqn](https://latex.codecogs.com/svg.latex?y_i) = 0, 1 for negative and positive examples respectively.n is the number of training examples.

* Initialize weights ![eqn](https://latex.codecogs.com/png.latex?w_%7B1%2Ci%7D%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%2C%20%5Cfrac%7B1%7D%7B2l%7D)  for  ![eqn](https://latex.codecogs.com/svg.latex?y_i). m and l are the number of negatives and positives respectively.

* for t(iteration) = 1,...,T :
    1. Normalize the weights to make it as probability distribution. 
    ![eqn](https://latex.codecogs.com/png.latex?w_%7Bt%2Ci%7D%5Cleftarrow%20%5Cfrac%7Bw_%7Bt%2Ci%7D%7D%7B%5Csum__%7Bj%3D1%7D%5En%20w_%7Bt%2Cj%7D%7D)
    
    1. For each j, train a classifier (weak learner). The error is evaluated with respect to weights for every record in that iteration.
    ![eqn](https://latex.codecogs.com/png.latex?%5Cepsilon_j%20%3D%20%5Csum_i%20w_i%7Ch_j%28x_i%29-y_i%7C)
    
    1. Choose the best classifier (with lowest error value).
    
    1. Update the weights:
    ![eqn](https://latex.codecogs.com/png.latex?w_%7Bt&plus;1%2Ci%7D%20%3Dw_%7Bt%2Ci%7D%5Cbeta_t%5E%7B1-e_1%7D)
    
    where ![eqn](https://latex.codecogs.com/png.latex?%5Cepsilon_i) = 0, if the example is classified correctly.
    ![eqn](https://latex.codecogs.com/png.latex?%5Cepsilon_i) = 1, if the example is not correctly classified.
    Beta is calculated as ![eqn](https://latex.codecogs.com/png.latex?%5Cbeta_t%20%3D%20%5Cfrac%7B%5Cepsilon_t%7D%7B1-%5Cepsilon_t%7D)
    
    1. The final strong classifier is:
    
    ![eqn](https://latex.codecogs.com/png.latex?h%28x%29%3D%20%5Cbegin%7Bcases%7D%201%26%5Csum_%7Bt%3D1%7D%20%5Calpha_th_t%28x%29%5Cgeq%20%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bt%3D1%7D%5ET%5Calpha_t%5C%5C%200%26Otherwise%20%5Cend%7Bcases%7D)
    
    where ![eqn](https://latex.codecogs.com/png.latex?%5Calpha_t%20%3D%20log%5Cfrac%7B1%7D%7B%5Cbeta_t%7D)
    
A very aggressive approach is taken to reduce the majority of features. 
Initial experiments demonstrated that, best 200 features accounts for 95% accuracy with a false positive rate of 1 in 14084.
Initial rectangle features constructed where easy to interpret.
For example, eyes and nose, which are darker with a bright region at the center.
And eyes region are darker than cheeks.(See figure 6).
  

**Figure 6**

![Haar features](https://miro.medium.com/max/754/1*Ghu_csPdGbAb5YvChZcBMQ.png "Figure 6")


#### Attentional Cascade:

A cascade of classifiers are created to reduce computation time by reducing number of sub-windows that are unlikely to be a frontal face.
The detection process is by creating sequence of AdaBoost classifiers that discards majority of sub-windows maintaining a low false positive rate.
Only sub-windows that pass through the first classifier triggers second classifiers, and the windows that pass through the second, trigger third.(See Figure 7)
The classifiers in the first stages are simple classifiers which gets complex with increase in stages.
Classifiers in each stages have thresholds adjusted to minimize false negatives coming out of the classifier.
A lower threshold in an AdaBoost classifier yields higher detection rates and higher false positive rates.
As the false positives can be discarded on further stages, it is important to maintain low false negative in any stage of the cascade.
Also the classifier at early stages tries to discard as much sub-windows as possible.
The higher the number of sub-windows pass through, the harder it becomes to detect and take more computations.
The more the number of windows passing through early classifiers push ROC cure downwards(High false positive rates).


For example, a simple two feature classifier on the first stage(See figure 6), can detect 100% of the faces(true positives) and 40% of false positives(which can be filtered on further stages) by reducing the threshold to minimum.
The authors by experiments found computation of the two-feature classifier amounts to about 60 microprocessor instructions which is comparatively much lower than a single layer perceptron.

#### Training cascades of classifiers:
Training of cascades of classifiers involves trade-offs between computational time and accuracy.
The number of features is directly proportional to both Detection rates and computational time.
So, it is important to optimize

1. Number of stages.
1. Number of features in each stage.
1. The threshold value for classifiers in each stage.

The authors made a framework to define these values using minimum reduction in false positive and maximum decrease in detections(True positives) by testing them on a validation set.
The authors concluded with a detection cascade of **38 stages** and over **6000 features**.
On an average, a sub-window undergoes 10 feature evaluation.
And it is 15 times faster than the previous best that time.


**Figure 7**

![cascade classifier](https://miro.medium.com/max/600/0*O4k602EVv7smbTFG "Figure 7")

#### Image processing:
Since the training sub-windows are normalized, the input sub-windows should be normalized as well.
The variance can be calculated using : 

![eqn](https://latex.codecogs.com/png.latex?%5Csigma%5E2%20%3D%20m%5E2%20-%20%5Cfrac%7B1%7D%7BN%7D%5Csum%20x%5E2)

m --> mean,
x --> pixel value 

both calculated from integral image.

#### Scanning:

The final detector scans across image at multiple scales and locations.
Scaling here indicates scaling the features and not the image. 
Experimentally, the value of scaling factor is set to 1.25.

Scanning in different locations is achieved by shifting some pixels in a sub-window.
Calculating shifting factor given the scaling factor (s) : ![eqn](https://latex.codecogs.com/png.latex?%5Bs%5CDelta%5D)

[] --> rounding operation.

It is found that value of shifting factor affect computation speed and accuracy.
Higher the factor, lesser the computational time and lesser the accuracy since higher shifting factor leads to less sub-window created per image.

Here the results are shown for the Delta value of 1.0

#### Handling multiple detections:
It is important for a detection to return a single detection per face.
To process the multiple detections overlapping each other, simply by taking an average of all the boxes.

For example,

The top left corner of the bounding box is the average of top left corners of all overlapping boxes.

#### Experiments on dataset:
The system is tested on MIT+CMU.

To create a ROC curve threshold of the final classifiers in adjusted from ![eqn](https://latex.codecogs.com/png.latex?-%5Cinfty%20%5C%20to%5C%20&plus;%5Cinfty).

While adjusting to +infinity, the detection rate becomes 0. Meaning the all the sub-windows are rejected.

While adjusting the threshold to -infinity, the detection rate becomes 1. Meaning all the sub-windows are passed.
Which is equal to removal of the layer itself.

To calculate respective detection rates, simply divide by the number of sub-windows scanned.
In this case, it is 75,081,800 sub-windows scanned.

  

## Results
A cascaded classifier with **38 stages, 6061 features**, trained with **9832** training images i.e. 4916 unique images * 2(mirrored) and 10,000 non-face sub-windows.
Training images are scaled and aligned to 24*24 pixels.

An average of *10 feature* is evaluated per sub-window.
On a 700 Mhz Pentium III processor, the face detector can process a 384 by 288 pixel image in about .067 seconds.
This is 15 time faster than the previous best at that time.

 
## What does the future hold?
Higher detection rate can be achieved by giving image differences in video sequences or by giving in color images which can provide more information rather than a single gray scale image used in this paper.

#### Limitations
The classifier is sensitive to slight changes in orientation of head either vertical or horizontal and to brightness.


