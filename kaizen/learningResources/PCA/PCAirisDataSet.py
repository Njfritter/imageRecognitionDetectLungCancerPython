#!/usr/bin/env python
#####################


import pandas as pd

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

from sklearn.preprocessing import StandardScaler

import numpy as np
##########################################

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()

X = df.ix[:,0:4].values
y = df.ix[:,4].values

#the iris data set is stored in the form of a 150x4 matrix
#columns are different features, each row represents a different sample	
#each row can be viewed as a 4 dimensional vector (_,_,_,_

traces = []

legend = {0:False, 1:False, 2:False, 3:True}

colors = {'Iris-setosa': 'rgb(31, 119, 180)', 
          'Iris-versicolor': 'rgb(255, 127, 14)', 
          'Iris-virginica': 'rgb(44, 160, 44)'}

for col in range(4):
    for key in colors:
        traces.append(Histogram(x=X[y==key, col], 
                        opacity=0.75,
                        xaxis='x%s' %(col+1),
                        marker=Marker(color=colors[key]),
                        name=key,
                        showlegend=legend[col]))

data = Data(traces)

layout = Layout(barmode='overlay',
                xaxis=XAxis(domain=[0, 0.25], title='sepal length (cm)'),
                xaxis2=XAxis(domain=[0.3, 0.5], title='sepal width (cm)'),
                xaxis3=XAxis(domain=[0.55, 0.75], title='petal length (cm)'),
                xaxis4=XAxis(domain=[0.8, 1], title='petal width (cm)'),
                yaxis=YAxis(title='count'),
                title='Distribution of the different Iris flower features')

fig = Figure(data=data, layout=layout)
py.plot(fig) #uncomment to plot the distribution of the iris data set

X_std = StandardScaler().fit_transform(X)

#Below is a more verbose way to create the covariance matrix that shows the specifics of calculating it
#mean_vec = np.mean(X_std, axis=0)
#cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#print('Covariance matrix \n%s' %cov_mat)

#i believe this represents the covariance of the 4 features, as in how the 4 features change with respetct
#to each other
print("----------------------------------------------------")
print("Using standardized data")
print('\nNumPy covariance matrix: \n%s' %np.cov(X_std.T))

#eigenvectors/values of cov_mat describe the principle components/magnitudes of the subspace we are projecting the data to


cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

print("----------------------------------------------------")

#the correlation matrix can be understood as the normalized covariance matrix.

cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)
print("Eigendecomposition of the standardized data based on the correlation matrix:")
print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
print("----------------------------------------------------")

cor_mat2 = np.corrcoef(X.T)

print("Eigendecomposition of the raw data based on the correlation matrix:")
print('\nNumPy correlation matrix: \n%s' %cor_mat2)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
print("----------------------------------------------------")

#singular value decomposition
u,s,v = np.linalg.svd(X_std.T)

#the eignevectors form the axes (basis vectors) for the subspace of lower dimension that we are projecting the data onto
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#The eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

trace1 = Bar(
        x=['PC %s' %i for i in range(1,5)],
        y=var_exp,
        showlegend=False)

trace2 = Scatter(
        x=['PC %s' %i for i in range(1,5)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = Data([trace1, trace2])

layout=Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = Figure(data=data, layout=layout)
py.plot(fig) #this plot shows how much variance can be expressed by each eigenvector (principle component)


matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)


#In this last step we will use the 4×24×2-dimensional projection matrix WW to transform our samples onto the new subspace via the equation
#Y=X×WY=X×W, where YY is a 150×2150×2 matrix of our transformed samples.

Y = X_std.dot(matrix_w)
traces = []

for name in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):

    trace = Scatter(
        x=Y[y==name,0],
        y=Y[y==name,1],
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)


data = Data(traces)
layout = Layout(showlegend=True,
                scene=Scene(xaxis=XAxis(title='PC1'),
                yaxis=YAxis(title='PC2'),))

fig = Figure(data=data, layout=layout)
py.plot(fig)

print("So what we've done is project samples in 4 dimensions to a 2 dimensional subspace that\n we can actually visualize, which is cool")
