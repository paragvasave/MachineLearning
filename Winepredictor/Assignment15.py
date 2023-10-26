
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def WinePredictor():
    # Load dataset
    wine = load_wine()
    
    # print names of the feature 
    print(wine.feature_names)

    # print target names of the feature
    print(wine.target_names)

    # print target column
    print(wine.target)

    # split dataset
    X_Train,X_test,Y_Train,Y_Test=train_test_split(wine.data,wine.target,test_size=0.3)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_Train,Y_Train)

    prediction=model.predict(X_test)
    Accuracy = accuracy_score(Y_Test,prediction)

    print("Accuracy is :",Accuracy)


def main():
    print("--------------------- Application for Wine Predictor ---------------------")
    print("______________________ KNN ______________________")
    WinePredictor()

if __name__=="__main__":
    main()