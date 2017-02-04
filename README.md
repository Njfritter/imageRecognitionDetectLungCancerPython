# imageRecognitionDetectLungCancerPython

## TABLE OF CONTENTS

* [Abstract](#Abstract)
* [Additional Resources](#Additional-Resources)
* [Picking The Algorithm](#Picking-The-Algorithm)
* [Dependencies Required](#Dependencies-Required)
* [Steps Required](#Steps-Required)
* [The Findings](#And-now-the-Analysis)
* [Methodology](#Method)
* [General Findings](#General-Findings])

## ABSTRACT

 - This repo is dedicated to using data science methods in Python to analyze low-dose CT images of patient lungs to attempt to determine whether cancer will develop within a year.
 - This data comes from the annual [Kaggle competition](https://www.kaggle.com/c/data-science-bowl-2017/data) that took place in 2017 and is linked
 - We use various machine learning methods such as K-Nearest Neighbors, Neural Networks, Principal Component Analysis and Decision Trees/Random Forests. We will look closely at each of these methods and choose the best method based on accuracy + insight into the relationships between expalnatory variables 
 - First we sample these methods on one of the most used, well-known set of images out there: The MNIST data set. Each method is used and analyzed for effectiveness (currently the KNN code is not working)
 - After we dip our feet in each of these methods, we create a database to extract as much data as we choose (the data set is 66.88 GB after all...) and then go HAM on Machine Learning!!!

## ADDITIONAL RESOURCES

We used the following documentation to further educate ourselves through the process of this project.

 - The MNIST program was built using the following resources:

 - [Principal Component Analysis](http://austingwalters.com/pca-principal-component-analysis/)

 - [Neural Networks](http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py)

 - 



## Introduction
-This project originated from the UCSB Project Group in Winter 2017. The members that contributed to this are 