import pandas as pd
from sklearn.linear_model import LinearRegression

def HeadBrainPredictor():
    # Load dataset
    Dataset = pd.read_csv("HeadBrainDataset.csv")
    X = Dataset['Head Size(cm^3)'].values
    Y = Dataset['Brain Weight(grams)'].values

    X = X.reshape((-1,1))

    n = len(X)

    reg = LinearRegression()

    reg = reg.fit(X,Y)

    y_pred = reg.predict(X)

    r2 = reg.score(X,Y)
    print("Value of RSquare : ",r2)

def main():
    print("Supervised Machine Learning")
    print("Linear Regression on Head and Brain size dataset")

    HeadBrainPredictor()

if __name__=="__main__":
    main()