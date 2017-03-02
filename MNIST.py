# We will be doing two methods of analysis on the handwritten data set: 
# Principal Component Analysis
# AND
# Neural Networks


#######
######
#####
####
###
##
# FIRST SECTION: GATHER DATA
##
###
####
#####
######
#######

# First let's get the data
# And import the module from scikit learn that will do this
import sys
from sklearn.utils import shuffle
from sklearn import datasets
print("Fetching Data Remotely")
mnist = datasets.fetch_mldata("MNIST Original")

# Next we import the matplotlib library as it is used in every function
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt




#######
######
#####
####
###
##
# NEXT SECTION: PCA ANALYSIS
##
###
####
#####
######
#######


# Here we will perform PCA to reduce dimensionality 
# Courtesy of http://austingwalters.com/pca-principal-component-analysis/
# This is an extra method to use Singular Value Decomposition (SVD)
"""
import numpy as np
matrix = np.matrix(data) * np.matrix(data).transpose() 
leftSingular, rightSingular, nonSingular = np.linalg.svd(matrix) 
scoreMatrix = leftSingular * rightSingular
"""

def princicalComponentAnalysis():
	# https://gist.github.com/mrgloom/6622175
	# Explanation: https://lazyprogrammer.me/tutorial-principal-components-analysis-pca/
	from sklearn.decomposition import PCA

	X, y = mnist.data / 255., mnist.target
	X_train, X_test = X[:60000], X[60000:]
	y_train, y_test = y[:60000], y[60000:]

	#X_train, y_train = shuffle(X_train, y_train)
	#X_train, y_train = X_train[:1000], y_train[:1000]  # lets subsample a bit for a first impression


	pca = PCA(n_components = 2, svd_solver = 'randomized')
	#pca = PCA(n_components = 2)

	fig, plot = plt.subplots()
	fig.set_size_inches(50, 50)
	plt.prism()

	X_transformed = pca.fit_transform(X_train)
	print(pca.explained_variance_ratio_)
	plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c = y_train)
	plot.set_xticks(())
	plot.set_yticks(())

	plt.tight_layout()
	plt.show()
	#plt.savefig("mnist_pca.png")


#######
######
#####
####
###
##
# NEXT SECTION: NEURAL NETWORK ANALYSIS
##
###
####
#####
######
#######

def neuralNetwork():
	# This project was taken from 
	# http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py
	# We will use this code to classify handwritten digits from 0 - 9
	# Here we use a Neural Network
	# First load the correct packages
	from sklearn.neural_network import MLPClassifier

	# rescale the data, use the traditional train/test split
	X, y = mnist.data / 255., mnist.target
	X_train, X_test = X[:60000], X[60000:]
	y_train, y_test = y[:60000], y[60000:]

	# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
	#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
	mlp = MLPClassifier(hidden_layer_sizes=(100,100), 
						max_iter=10, 
						alpha=1e-4,
	                    solver='sgd', 
	                    verbose=10, 
	                    tol=1e-4, 
	                    random_state=1,
	                    learning_rate_init=.1)

	mlp.fit(X_train, y_train)
	print("Training set score: %f" % mlp.score(X_train, y_train))
	print("Test set score: %f" % mlp.score(X_test, y_test))

	fig, axes = plt.subplots(4, 4)
	# use global min / max to ensure all weights are shown on the same scale
	vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
	for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
	    ax.matshow(coef.reshape(28, 28), 
	    			cmap=plt.cm.gray, 
	    			vmin=.5 * vmin,
	               	vmax=.5 * vmax)

	    ax.set_xticks(())
	    ax.set_yticks(())

	plt.show()


#######
######
#####
####
###
##
# NEXT SECTION: K NEAREST NEIGHBORS ANALYSIS
##
###
####
#####
######
#######

def kNearestNeighbors():
	# Will potentially build from scratch in the future from below links
	# https://lazyprogrammer.me/tutorial-k-nearest-neighbor-classifier-for-mnist/
	# http://andrew.gibiansky.com/blog/machine-learning/k-nearest-neighbors-simplest-machine-learning/
	import random
	from sklearn.neighbors import KNeighborsClassifier

	X, y = mnist.data / 255., mnist.target
	X_train, X_test = X[:60000], X[60000:]
	y_train, y_test = y[:60000], y[60000:]


	KNN = KNeighborsClassifier(n_neighbors=5)
	KNN.fit(X_train, y_train)

	KNN_pred = KNN.predict(X_test)
	
	# Cross Validation Results Exercise 3.3 for Decision Tree
	scores = cross_validation.cross_val_score(clf, X_train, y_train.tolist(), cv = 5)
	print(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))



def decisionTree():
	# https://github.com/efebozkir/handwrittendigit-recognition/blob/master/decisiontreefile.py
	from sklearn import tree
	from sklearn import metrics
	from sklearn import cross_validation

	X, y = mnist.data / 255., mnist.target
	X_train, X_test = X[:6000], X[6000:]
	y_train, y_test = y[:6000], y[6000:]

	trainingImagesCount = len(X_train)
	testingImagesCount = len(y_train)

	clf = tree.DecisionTreeClassifier(criterion = "gini",
											max_depth = 32, 
											max_features = 784)
	#clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, 
				y_train)
	#clf = clf.fit(trainingImages[:60000], trainingLabels[:60000])

	predictionRes = clf.predict(X_test)

	

	# Cross Validation Results Exercise 3.3 for Decision Tree
	scores = cross_validation.cross_val_score(clf, X_train, y_train.tolist(), cv = 5)
	print(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

	# Pixel importances on 28*28 image
	importances = clf.feature_importances_
	importances = importances.reshape((28, 28))

	# Plot pixel importances
	plt.matshow(importances, cmap = plt.cm.hot)
	plt.title("Pixel importances for decision tree")
	plt.show()

	"""
	# Decision Tree as output -> decision_tree.png
	dot_data = StringIO.StringIO()
	tree.export_graphviz(clf, out_file=dot_data)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_png('decision_tree.png')

	# IMPORTANT NOTE: If you change the number of training images, you should also change the number of images
	# in cross validation.

	# decision_tree.png can be huge. Please zoom in to see the tree more clearly.
	Contact GitHub API Training Shop Blog About
	"""


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == 'NN':
            neuralNetwork()
        elif sys.argv[1] == 'PCA':
            princicalComponentAnalysis()
        elif sys.argv[1] == 'KNN':
            kNearestNeighbors()
        elif sys.argv[1] == 'DT':
            decisionTree()
