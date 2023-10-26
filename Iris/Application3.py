# Supervised Machine Learning
# Iris dataset with Decision Tree

# In this application we remove one entry from each label of iris dataset and train with the remaining entries.
# We apply predictions based on Decision tree with that removed entries.

# Consider below charachteristics of machine learning application.
# Classifier : Decision tree
# Dataset :             Iris Dataset
# Features :            Sepal width, Sepal Length, Petal width,petal length
# Labels :              Versicolor, Setosa, Virginica
# Training dataset :    147 entries
# testing dataset :     3 entries

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

print("Feature names of iris data set : ",iris.feature_names)

print("Target names of iris data set : ",iris.target_names)

# Indices of removed elements
test_index = [1,51,101]


# Training data with removed elements
train_data = np.delete(iris.data,test_index,axis=0)                 # x1
train_target = np.delete(iris.target,test_index)                    # y1


# Testing on training data
test_data = iris.data[test_index]                                   # x2
test_target = iris.target[test_index]                               # y2


# form decision tree classifier
classifier = tree.DecisionTreeClassifier()

# Apply on training data to form tree
classifier.fit(train_data,train_target)

print("Values that we removed for testing : ",test_target)
print("Result of testing : ",classifier.predict(test_data))
