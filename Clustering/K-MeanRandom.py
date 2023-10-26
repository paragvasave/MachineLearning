# In this case study we are generating the data set at run time randomly and applying user defined K-Mean algorithm.
import pandas as pd
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def KMeanRandom():
    # Set three centers, the model should predict similar results
    center_1 = np.array([1,1])
    print(center_1)
    print("----------------------------------------------")

    center_2 = np.array([5,5])
    print(center_2)
    print("----------------------------------------------")

    center_3 = np.array([8,1])
    print(center_3)
    print("----------------------------------------------")

    # Generating random data and center it to three centers
    data_1 = np.random.randn(7,2) + center_1
    print("Elements of first cluster with size : "+(str(len(data_1))))
    print(data_1)
    print("----------------------------------------------")

    data_2 = np.random.randn(7,2) + center_2
    print("Elements of second cluster with size : "+(str(len(data_2))))
    print(data_2)
    print("----------------------------------------------")

    data_3 = np.random.randn(7,2) + center_3
    print("Elements of third cluster with size : "+(str(len(data_3))))
    print(data_3)
    print("----------------------------------------------")

    data = np.concatenate((data_1,data_2,data_3),axis=0)
    print("Size of complete data set "+str(len(data)))
    print(data)
    print("----------------------------------------------")

    plt.scatter(data[:,0],data[:,1],s=7)
    plt.title('Input Dataset')
    plt.show()
    print("----------------------------------------------")
    # Number of clusters
    k=3

    # Number of training data
    n = data.shape[0]
    print("Total number of elements are : ",n)
    print("----------------------------------------------")
    
    #Number of features in the data
    c = data.shape[1]
    print("Total number of features are : ",c)
    print("----------------------------------------------")

    # Generate random centers, here we use sigma and mean to ensure it represent the whole data
    mean = np.mean(data,axis=0)
    print("Value of mean : ",mean)
    print("----------------------------------------------")

    # Calculate standard deviation
    std = np.std(data,axis=0)
    print("Values of std : ",std)
    print("----------------------------------------------")

    centers = np.random.randn(k,c)*std + mean
    print("Random points are : ",centers)
    print("----------------------------------------------")

    # Plot the data and the centers generated as random
    plt.scatter(data[:,0],data[:,1],c='r',s=7)
    plt.scatter(centers[:,0],centers[:,1],marker='*',c='g',s=150)
    plt.title('Input Dataset with random sentroid *')
    plt.show()
    print("----------------------------------------------")

    centers_old = np.zeros(centers.shape)       # to store old centers
    centers_new = deepcopy(centers)             # Store new centers

    print("Values of old centroids")
    print(centers_old)
    print("----------------------------------------------")

    print("Values of new centroids")
    print(centers_new)
    print("----------------------------------------------")

    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n,k))

    print("Initial distances are ")
    print(distances)
    print("----------------------------------------------")

    error = np.linalg.norm(centers_new-centers_old)
    print("values of error is : ",error)

    # When, after an update, the estimate of that center stays the same, exit loop

    while error !=0:
        print("Value of error is  : ",error)
        # Measure the distance to every center
        print("Measure the distance to every center")
        for i in range(k):
            print("Iteration number : ",i)
            distances[:,i] = np.linalg.norm(data - centers[i],axis=1)
        
        # Assign all training data to closest center
        clusters = np.argmin(distances,axis=1)

        centers_old = deepcopy(centers_new)

        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i],axis = 0)
        error = np.linalg.norm(centers_new-centers_old)
    # end of while
    centers_new

    # Plot the data and the centers generated as random
    plt.scatter(data[:,0],data[:,1],s=7)
    plt.scatter(centers_new[:,0],centers_new[:,1],marker='*',c='g',s=150)
    plt.title("Final data with Centroid")
    plt.show()


def main():
    print("----------------- Unsupervised Machine Learning -----------------")
    print("Clustering using K Mean Algorithm")
    print("-----------------------------------------------------------------")

    KMeanRandom()

if __name__=="__main__":
    main()