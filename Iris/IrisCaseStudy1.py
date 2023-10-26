from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def MarvellousKNeighborsClassifier():
    Dataset = load_iris()               # load the data set

    Data = Dataset.data                 
    Target = Dataset.target

    # Division A B C D quadrant
    Data_train,Data_test,Target_train,Target_test = train_test_split(Data,Target,test_size=0.5)

    # Select the algorithm
    Classifier = KNeighborsClassifier()

    # Train the model or build the model
    Classifier.fit(Data_train,Target_train)

    # Test the model
    Prediction = Classifier.predict(Data_test)
    
    Accuracy = accuracy_score(Target_test,Prediction)

    return Accuracy


def main():
    Ret = MarvellousKNeighborsClassifier()
    print("Accuracy of iris dataset with KNN : ",Ret*100)

if __name__=="__main__":
    main()