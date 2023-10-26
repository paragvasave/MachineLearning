from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance


def euc(a,b):
    return distance.euclidean(a,b)

class KNNX():
    def fit(self,trainingdata,trainingtarget):
        self.TrainingData= trainingdata
        self.TrainingTarget = trainingtarget
        
    def predict(self,Testing_Data):
        PredictionList=[]
        for everyRow in Testing_Data:
            Result=self.closest(everyRow)
            PredictionList.append(Result)
        
        return PredictionList
    
    def closest(self,Row):
        minimumDistance = euc(Row,self.TrainingData[0])
        minimumIndex = 0

        for i in range(1,len(self.TrainingData)):
            Distance = euc(Row,self.TrainingData[i])
            if(Distance<minimumDistance):
                minimumDistance = Distance
                minimumIndex = i
        return self.TrainingTarget[minimumIndex]


def Model():
    Dataset = load_iris()
    Data = Dataset.data
    Target = Dataset.target

    Data_Train,Data_Test,Target_Train,Target_Test = train_test_split(Data,Target,test_size=0.25)
    Classifier = KNNX()

    Classifier.fit(Data_Train,Target_Train)
    Predictions =Classifier.predict(Data_Test)

    Accuracy = accuracy_score(Target_Test,Predictions)

    return Accuracy


def main():
    Ret = Model()
    print("Accuracy : ",Ret*100)

if __name__=="__main__":
    main()