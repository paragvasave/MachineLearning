# Application 10 : 
# Supervised Machine Learning : Linear Regression with user defined algorithm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def HeadBrainPredictor():
    # Load dataset
    Dataset = pd.read_csv('HeadBrainDataset.csv')

    print("Size of dataset : ",Dataset.shape)

    X = Dataset['Head Size(cm^3)'].values
    Y = Dataset['Brain Weight(grams)'].values

    # Least Square method
    X_Bar = np.mean(X)              # X mean
    Y_Bar = np.mean(Y)              # Y mean

    n =len(X)
    
    numerator = 0
    denominator = 0

    # Equation of line Y= mX+c

    for i in range(n):
        numerator += (X[i]-X_Bar)*(Y[i]-Y_Bar)
        denominator += (X[i]-X_Bar)**2

    m = numerator/denominator       # m = sum((X-Xbar)*(Y-Ybar)) / sum(X-Xbar)^2

    C = Y_Bar-(m*X_Bar)             # Y = mX+C

    print("Slope of regression line : ",m)
    print("Y intercept of regression line : ",C)

    X_Max = np.max(X)+100
    X_Min = np.min(X)-100


    # Display plotting of above points
    x=np.linspace(X_Min,X_Max,n)

    y = C + m*x

    plt.plot(x,y,color='#58b970',label='Regression Line')

    plt.scatter(X,Y,color='#ef5423',label='scatter plot')

    plt.xlabel('Head size in cm^3')

    plt.ylabel("Brain weight in gram")

    plt.legend()

    plt.show()


def main():
    print("Supervised machine learing.")
    print("Linear regression on Head and Brain size data set.")

    HeadBrainPredictor()

if __name__=="__main__":
    main()