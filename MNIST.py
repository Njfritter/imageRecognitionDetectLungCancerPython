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
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
print("Fetching Data Remotely")
mnist = datasets.fetch_mldata("MNIST Original")



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
	import matplotlib as mpl
	mpl.use('TkAgg')
	import matplotlib.pyplot as plt
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
	plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train)
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
	import matplotlib as mpl
	mpl.use('TkAgg')
	import matplotlib.pyplot as plt
	from sklearn.neural_network import MLPClassifier

	# rescale the data, use the traditional train/test split
	X, y = mnist.data / 255., mnist.target
	X_train, X_test = X[:60000], X[60000:]
	y_train, y_test = y[:60000], y[60000:]

	# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
	#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
	mlp = MLPClassifier(hidden_layer_sizes=(100,), 
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
	# https://lazyprogrammer.me/tutorial-k-nearest-neighbor-classifier-for-mnist/
	# http://andrew.gibiansky.com/blog/machine-learning/k-nearest-neighbors-simplest-machine-learning/
	def predict(self, point):
    # We have to copy the data set list, because once we've located the best
    # candidate from it, we don't want to see that candidate again, so we'll delete it.
	    candidates = self.dataset[:]
	    
	    # Loop until we've gotten all the neighbors we want.
	    neighbors = []
	    while len(neighbors) < self.k:
	        # Compute distances to every candidate.
	        distances = [self.distance(x[0], point) for x in candidates]
	        
	        # Find the minimum distance neighbor.
	        best_distance = min(distances)
	        index = distances.index(best_distance)
	        neighbors.append(candidates[index])
	        
	        # Remove the neighbor from the candidates list.
	        del candidates[index]
	    
	    # Predict by averaging the closets k elements.
	    prediction = self.consensus([value[1] for value in neighbors])
	    return prediction

	# Get the figure and axes.
	fig, axes = plt.subplots(5, 5)
	axes = axes.reshape(25)
	fig.suptitle("Random Sampling of MNIST")

	# Plot random images.
	indices = random.randint(len(train_images), size=25)
	for axis, index in zip(axes, indices):
	    image = train_images[index, :, :]
	    axis.get_xaxis().set_visible(False)
	    axis.get_yaxis().set_visible(False)
	    axis.imshow(image, cmap = cm.Greys_r)

	# Let's go ahead and replace the old implementation.
	NearestNeighborClassifier.predict = predict

	def euclidean_distance(img1, img2):
	    # Since we're using NumPy arrays, all our operations are automatically vectorized.
	    # A breakdown of this expression:
	    #     img1 - img2 is the pixel-wise difference between the images
	    #     (img1 - img2) ** 2 is the same thing, with each value squared
	    #     sum((img1 - img2) ** 2) is the sum of the elements in the matrix.
	    return sum((img1 - img2) ** 2)

	from collections import defaultdict
	def get_majority(votes):
	    # For convenience, we're going to use a defaultdict.
	    # This is just a dictionary where values are initialized to zero
	    # if they don't exist.
	    counter = defaultdict(int)
	    for vote in votes:
	        # If this weren't a defaultdict, this would error on new vote values.
	        counter[vote] += 1
	    
	    # Find out who was the majority.
	    majority_count = max(counter.values())
	    for key, value in counter.items():
	        if value == majority_count:
	            return key

	# Create the predictor class.
	class MNISTPredictor(NearestNeighborClassifier):
	    def distance(self, p1, p2):
	        return euclidean_distance(p1, p2)
	    
	    def consensus(self, values):
	        return get_majority(values)
	    
	# Convert our data set into an easy format to use.
	# This is a list of (x, y) pairs. x is an image, y is a label.
	dataset = []
	for i in xrange(len(train_images)):
	    dataset.append((train_images[i, :, :], train_labels[i]))
	    
	# Create a predictor for various values of k.
	ks = [1, 2, 3, 4, 5, 6]
	predictors = [MNISTPredictor(dataset, k) for k in ks] 


	def predict_test_set(predictor, test_set):
	    """Compute the prediction for every element of the test set."""
	    predictions = [predictor.predict(test_set[i, :, :]) 
	                   for i in xrange(len(test_set))]
	    return predictions

	# Choose a subset of the test set. Otherwise this will never finish.
	test_set = test_images[0:100, :, :]
	all_predictions = [predict_test_set(predictor, test_set) for predictor in predictors]

	def evaluate_prediction(predictions, answers):
	    """Compute how many were identical in the answers and predictions,
	    and divide this by the number of predictions to get a percentage."""
	    correct = sum(asarray(predictions) == asarray(answers))
	    total = float(prod(answers.shape))
	    return correct / total

	labels = asarray(test_labels[0:100])
	accuracies = [evaluate_prediction(pred, labels) for pred in all_predictions]

	# Draw the figure.
	fig = figure(1)
	plt.plot(ks, accuracies, 'ro', figure=fig)

	fig.suptitle("Nearest Neighbor Classifier Accuracies")
	fig.axes[0].set_xlabel("k (# of neighbors considered)")
	fig.axes[0].set_ylabel("accuracy (% correct)");
	fig.axes[0].axis([0, max(ks) + 1, 0, 1]);


def decisionTree():
	# https://github.com/efebozkir/handwrittendigit-recognition/blob/master/decisiontreefile.py
	import matplotlib as mpl
	mpl.use('TkAgg')
	import matplotlib.pyplot as plt
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

	print(metrics.classification_report(y_test.tolist(), predictionRes, digits = 4))

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
