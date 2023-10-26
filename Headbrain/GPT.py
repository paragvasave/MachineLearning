import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def linear_regression(X, Y):
    # Calculate the means of X and Y
    X_bar = np.mean(X)
    Y_bar = np.mean(Y)

    n = len(X)

    numerator = 0
    denominator = 0

    # Calculate the coefficients of the linear regression equation (m and C)
    for i in range(n):
        numerator += (X[i] - X_bar) * (Y[i] - Y_bar)
        denominator += (X[i] - X_bar) ** 2

    m = numerator / denominator
    C = Y_bar - (m * X_bar)

    print(numerator,denominator)

    return m, C

def plot_regression_line(X, Y, m, C):
    # Calculate the predicted Y values
    Y_pred = m * X + C

    # Plot the regression line and scatter plot
    plt.scatter(X, Y, color='#ef5423', label='Scatter Plot')
    plt.plot(X, Y_pred, color='#58b970', label='Regression Line')

    plt.xlabel('Head size in cm^3')
    plt.ylabel('Brain weight in grams')
    plt.title('Linear Regression: Head Size vs. Brain Weight')

    plt.legend()
    plt.show()

def main():
    # Load the dataset
    dataset = pd.read_csv('HeadBrainDataset.csv')

    X = dataset['Head Size(cm^3)'].values
    Y = dataset['Brain Weight(grams)'].values

    # Perform linear regression
    m, C = linear_regression(X, Y)

    print("Slope of regression line:", m)
    print("Y intercept of regression line:", C)

    # Plot the regression line and scatter plot
    plot_regression_line(X, Y, m, C)

if __name__ == "__main__":
    main()
