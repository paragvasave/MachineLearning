from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def MarvellousDecisionTreeClassifier():
    Dataset = load_iris()

    Data = Dataset.data
    Target = Dataset.target

    Data_Train,Data_Test,Target_Train,Target_Test = train_test_split(Data,Target,test_size=0.5)

    Classifier = DecisionTreeClassifier()

    Classifier.fit(Data_Train,Target_Train)

    Prediction = Classifier.predict(Data_Test)

    Accuracy = accuracy_score(Target_Test,Prediction)

    return Accuracy

def main():
    Ret = MarvellousDecisionTreeClassifier()
    print("Accuracy of iris case study using DTC : ",Ret*100)
    

if __name__=="__main__":
    main()
