import pandas as pd
from seaborn import countplot
from matplotlib.pyplot import figure,show
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def TitanicLogistic():
    # Step 1: Load data
    titanic_data = pd.read_csv("TitanicDataset.csv")

    print(titanic_data.head())
    print("Number of passangers : "+str(len(titanic_data)))

    # Step 2: Analyze data
    print("Visualisation : Survived and Non Survied passangers")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x= target).set_title("Survived and Non-Survived passangers.")
    show()

    print("Visualisation : Survived and Non Survied passangers based on gender")
    figure()
    countplot(data=titanic_data,x=target,hue='Sex').set_title("Survived and Non-Survived passangers based on gender.")
    show()

    print("Visualisation : Survived and Non Survied passangers based on Passanger class")
    figure()
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and Non-Survived passangers based on Passanger class.")
    show()
    
    print("Visualisation : Survived and Non Survied passangers based on Age")
    figure()
    titanic_data['Age'].plot.hist().set_title("Survived and Non Survied passangers based on Age")
    show()

    print("Visualisation : Survived and Non Survied passangers based on Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Visualisation : Survived and Non Survied passangers based on Fare")
    show()

    # Step 3: Data Cleaning
    # Drop unnecessary columns
    titanic_data.drop(["zero", "sibsp", "Parch", "Embarked"], axis=1, inplace=True)

    # Convert non-numeric columns to numeric using one-hot encoding
    titanic_data = pd.get_dummies(titanic_data, columns=["Sex", "Pclass"], drop_first=True)

    # Convert the target variable to numeric
    titanic_data["Survived"] = titanic_data["Survived"].astype(int)

    # Split the data into features (x) and target variable (y)
    x = titanic_data.drop("Survived", axis=1)
    y = titanic_data["Survived"]

    # Step 4: Data Training
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)
    LogModel = LogisticRegression(max_iter=1000)
    LogModel.fit(xtrain, ytrain)

    # Step 5: Data Testing
    prediction = LogModel.predict(xtest)

    # Step 6: Calculate accuracy
    print("Classification report of Logistic Regression is : ")
    print(classification_report(ytest, prediction))

    print("Confusion Matrix of Logistic Regression is : ")
    print(confusion_matrix(ytest, prediction))

    print("Accuracy of Logistic Regression is : ")
    print(accuracy_score(ytest, prediction))


def main():
    print("-------------------- Titanic Survival Case Study --------------------")
    TitanicLogistic()


if __name__ == "__main__":
    main()
