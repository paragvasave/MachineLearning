# Application 12 :
# Supervised machine learning : Logistic regression : Titanic survival case study
# Features : Passenger ID, Gender, Age, Fare, Class, etc

import pandas as pd

from seaborn import countplot

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def TitanicLogistic():
    # Step 1 : Load data
    titanic_data = pd.read_csv("TitanicDataset.csv")
    
    # print(titanic_data.head())
    #print("Number of passangers : "+str(len(titanic_data)))

    # Step 2: Analyze data
    
    # print("Visualisation : Survived and Non Survied passangers")
    # figure()
    # target = "Survived"
    # countplot(data=titanic_data,x= target).set_title("Survived and Non-Survived passangers.")
    # show()

    # print("Visualisation : Survived and Non Survied passangers based on gender")
    # figure()
    # countplot(data=titanic_data,x=target,hue='Sex').set_title("Survived and Non-Survived passangers based on gender.")
    # show()

    # print("Visualisation : Survived and Non Survied passangers based on Passanger class")
    # figure()
    # countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and Non-Survived passangers based on Passanger class.")
    # show()
    
    # print("Visualisation : Survived and Non Survied passangers based on Age")
    # figure()
    # titanic_data['Age'].plot.hist().set_title("Survived and Non Survied passangers based on Age")
    # show()

    # print("Visualisation : Survived and Non Survied passangers based on Fare")
    # figure()
    # titanic_data["Fare"].plot.hist().set_title("Visualisation : Survived and Non Survied passangers based on Fare")
    # show()

    # Step 3: Data Cleaning
    titanic_data.drop("zero",axis=1,inplace=True)           # Removed zero column
    print(titanic_data.head())

    print("Value of Sex column")
    print(pd.get_dummies(titanic_data['Sex']))
    
    print("Value of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print(Sex.head(5))

    print("Value of Pclass column after removing one field.")
    Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(Pclass.head())

    print("Values of data set after concatenating new colmns")
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1)
    print(titanic_data.head())

    print("Values of data set after removing irrelevent columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head())

    # # Converting all the names of the column to str
    titanic_data.columns = [str(column) for column in titanic_data.columns]

    # # Converting all the values of the column to str
    for column in titanic_data.columns:
        titanic_data[column] = titanic_data[column].astype(str)

    x = titanic_data.drop("Survived",axis=1)
    y = titanic_data["Survived"]

    # Step 4 : Data Training
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)
    ytrain = ytrain.astype(str)

    LogModel = LogisticRegression(max_iter=1000)
    LogModel.fit(xtrain,ytrain)

    # Step 5 : Data Testing
    prediction = LogModel.predict(xtest)

    # Step 6 : Calculate accuracy
    print("Classification report of Logistic Regression is : ")
    print(classification_report(ytest,prediction))

    print("Confusion Matrix of Logistic Regression is : ")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of Logistic Regression is : ")
    print(accuracy_score(ytest,prediction))

def main():
    print("-------------------- Titanic Survival Case Study --------------------")

    TitanicLogistic()

if __name__=="__main__":
    main()