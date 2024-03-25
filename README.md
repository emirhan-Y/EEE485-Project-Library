# EEE485-Project-Library

Signature recognition using three different models. Planned models:

* KNN (K-Nearest Neighbors) (The optimal value of K will be investigated during testing)
* Logistic Regression (Both log-linear multiclass logistic regression and the set of independent binary logistic regression model types will be utilized)
* NN (Neural Network) (The type is yet to be determined, most likely a CNN, or a multi perceptron network)

## Authors 
- @emirhan-Y
- @Berkan5032

## Version History

### version 0
A test for linear regression and ridge regression. By utilizing the sklearn library, we tried to implement and optimize these models, using the sklearn models as our base line. 

### v0.1
An unstable implementation of KNN. All parameterized.
#### Changes
* KNN is added to the models
* Logger is added, to be used for runtime diagnostics
* Quick sort with trace auxillary list implementation for KNN
* Major restructuring of the libraries. Using __init__.py, all library modules are buried inside their own packages
#### Bugs
+ KNN has issues when data points are equally distant form the test point. In K=1 case, if there are two points which are closest to the test point, but their class is different, KNN chooses the last data point entered to the dataset, and disregards the other point. In K>1 cases, this is a bigger issue, as in the case where there are more than one Kth closest points, KNN chooses the last one again, which leads to incorrect results in some cases.

    + **fix:** Add a check in the end for equidistant points. If k equidistant points are present as the (K-m)th closest point to the test point, m < k, then form new point counts by dividing all k points by m/k, and add those as the count. For example, if K=6, and the closest 4 points were found without problem and say there are 2 Y=0 points and 2 Y=1 points. Say now we have 4 equidistant points remaining as the next closest points, with 2 of them being Y=0, one being Y=1, and the last being Y=2. Since we are looking for 2 more points, "squish" the four points into two points by weight: one is Y=0, a half is Y=1 and the last half is Y=2, now add these points to the K=6 points. We have 3 of Y=0, 2.5 of Y=1 and 0.5 of Y=2, hence the guess is Y=0. Note that I completely made this up, and there may be better ways to remove this problem.
    If for some case in the last K points, there are two or more modes, then those will be the guess, with equal probability. In the automatic case, K can be decreased or increased to break this equality, or we might simply accept both answers as "undecided."

### v0.2
* KNN model upgrade: In the cases of uncertainty, the model gives two or more answers (Will implement a proximity check system in the future).
* PCA Implementation: A rudimentary implementation of Principal Component Analysis. Dimension reduction capability was tested successfully.
* Data Preparation Tools: Tools for automatic data generation from images. Crops and resizes signatures images gathered from pdf files.
* Restructure: Some files were moved to their respective packages
* Added docstrings to majority of the classes, methods, files, and packages
* Docs fixes