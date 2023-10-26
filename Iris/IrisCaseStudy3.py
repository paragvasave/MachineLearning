from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class MarvellousKNeighborsClassifier:
    
    def fit(self,trainingdata,trainingtarget):
        self.TrainingData = trainingdata
        self.TrainingTarget = trainingtarget
 
    
    def predict(self,TestData):
        predictions = []
        for Value in TestData:
            result = self.closest(Value)
            predictions.append(result)
        
        return predictions


    def closest(self,row):
        minimumdistance = euc(row,self.TrainingData[0])
        minimumindex = 0

        for i in range(1,len(self.TrainingData)):
            Distance = euc(row,self.TrainingData[i])
            if Distance<minimumdistance:
                minimumdistance=Distance
                minimumindex=i

        return self.TrainingTarget[minimumindex]


def MarvellousML():
    Dataset = load_iris()
    Data = Dataset.data
    Target = Dataset.target

    # ACBD
    Data_Train,Data_Test,Target_Train,Target_Test = train_test_split(Data,Target,test_size = 0.5) # =0.3(Training 70%),(testing-30%)

    Classifier = MarvellousKNeighborsClassifier()

    Classifier.fit(Data_Train,Target_Train)

    Prediction = Classifier.predict(Data_Test)

    Accuracy = accuracy_score(Target_Test,Prediction)
    return Accuracy

def main():

    Ret = MarvellousML()
    print("Accuracy of Iris case study with KNN is : ",Ret*100)
    
if __name__=="__main__":
    main()