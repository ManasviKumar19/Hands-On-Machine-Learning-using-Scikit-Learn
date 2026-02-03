# Hands-on-Machine-Learning-using-Scikit-learn

This repository contains Jupyter Notebooks for learning MachineLearning using Scikit-learn.

## Chapter 2
In chapter 2  will work through an example project end to end, pretending to
be a recently hired data scientist at a real estate company. Here are the main steps you will go through:-
1. Look at the big picture.
2. Get the data.
3. Discover and visualize the data to gain insights.
4. Prepare the data for Machine Learning algorithms.
5. Select a model and train it.
6. Fine-tune your model.
7. Present your solution.
8. Launch, monitor, and maintain your system


## Chapter 3
In chapter 3 we will be using the MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labeled with the digit it represents. This set has been stud‐
ied so much that it is often called the “hello world” of Machine Learning: whenever people come up with a new classification algorithm they are curious to see how it will
perform on MNIST, and anyone who learns Machine Learning tackles this dataset sooner or later.


## Chapter 4
In chapter 4 we will start by looking at the Linear Regression model, one of the
simplest models there is. We will discuss two very different ways to train it:
• Using a direct “closed-form” equation that directly computes the model parame‐
ters that best fit the model to the training set (i.e., the model parameters that
minimize the cost function over the training set).
• Using an iterative optimization approach called Gradient Descent (GD) that
gradually tweaks the model parameters to minimize the cost function over the
training set, eventually converging to the same set of parameters as the first
method. We will look at a few variants of Gradient Descent that we will use again
and again when we study neural networks in Part II: Batch GD, Mini-batch GD,
and Stochastic GD.


Next we will look at Polynomial Regression, a more complex model that can fit non‐
linear datasets. Since this model has more parameters than Linear Regression, it is
more prone to overfitting the training data, so we will look at how to detect whether
or not this is the case using learning curves, and then we will look at several regulari‐
zation techniques that can reduce the risk of overfitting the training set.
Finally, we will look at two more models that are commonly used for classification
tasks: Logistic Regression and Softmax Regression.


## Chapter 5
A Support Vector Machine (SVM) is a powerful and versatile Machine Learning
model, capable of performing linear or nonlinear classification, regression, and even
outlier detection. It is one of the most popular models in Machine Learning, and any‐
one interested in Machine Learning should have it in their toolbox. SVMs are partic‐
ularly well suited for classification of complex small- or medium-sized datasets.
In chapter 5 will explain the core concepts of SVMs, how to use them, and how they
work.


## Chapter 6
Like SVMs, Decision Trees are versatile Machine Learning algorithms that can per‐
form both classification and regression tasks, and even multioutput tasks. They are
powerful algorithms, capable of fitting complex datasets. For example, in Chapter 2
you trained a DecisionTreeRegressor model on the California housing dataset, fit‐
ting it perfectly (actually, overfitting it).
In chapter 6 we will start by discussing how to train, visualize, and make predic‐
tions with Decision Trees. Then we will go through the CART training algorithm
used by Scikit-Learn, and we will discuss how to regularize trees and use them for
regression tasks. Finally, we will discuss some of the limitations of Decision Trees.


## Chapter 7
In chapter 7 we will discuss the most popular Ensemble methods, including bagging, boosting, and stacking. We will also explore Random Forests.


## Chapter 8
In chapter 8 we will discuss the curse of dimensionality and get a sense of what
goes on in high-dimensional space. Then, we will consider the two main approaches
to dimensionality reduction (projection and Manifold Learning), and we will go
through three of the most popular dimensionality reduction techniques: PCA, Kernel
PCA, and LLE.


## Chapter 9
In chapter 9 we will look at a few more unsupervised learning tasks
and algorithms:
### Clustering
The goal is to group similar instances together into clusters. Clustering is a great
tool for data analysis, customer segmentation, recommender systems, search
engines, image segmentation, semi-supervised learning, dimensionality reduc‐
tion, and more.
### Anomaly detection
The objective is to learn what “normal” data looks like, and then use that to
detect abnormal instances, such as defective items on a production line or a new
trend in a time series.
### Density estimation
This is the task of estimating the probability density function (PDF) of the random
process that generated the dataset. Density estimation is commonly used for
anomaly detection: instances located in very low-density regions are likely to be
anomalies. It is also useful for data analysis and visualization.
