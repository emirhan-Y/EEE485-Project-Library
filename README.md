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

### version KNN
A stable implementation of KNN. All parameterized.
#### Changes
* KNN is added to the models
* Logger is added, to be used for runtime diagnostics
* Quick sort with trace auxillary list implementation for KNN
* Major restructuring of the libraries. Using __init__.py, all library modules are buried inside their own packages