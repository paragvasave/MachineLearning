from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score

def euc(a,b):
    return distance.euclidean(a,b)

class Algorithm:
    def fit(self,trainingdata,trainingtarget):
        self.TrainingData = trainingdata
        self.TrainingTarget = trainingtarget
    
    def predict(self,TestingData):
        List_of_Predictions=[]

        for row in TestingData:
            result=self.closest(row)
            List_of_Predictions.append(result)
        
        return List_of_Predictions
    
    def closest(self,row):

        bestdistance = euc(self.TrainingData[0],row)
        print(bestdistance)
        bestindex = 0

        for i in range(1,len(self.TrainingData)):
            Distance = euc(row,self.TrainingData[i])
            if(Distance<bestdistance):
                bestdistance=Distance
                bestindex = i
                
        print(bestindex)
        return self.TrainingTarget[bestindex]



def MarvellousML():

    Dataset = load_iris()
    Data = Dataset.data
    Target = Dataset.target

    Data_Train,Data_Test,Target_Train,Target_Test=train_test_split(Data,Target,test_size=0.7)

    Classifier = Algorithm()
    Classifier.fit(Data_Train,Target_Train)

    Prediction = Classifier.predict(Data_Test)
    # Accuracy = accuracy_score(Target_Test,Prediction)

    # return Accuracy



def main():
    Ret = MarvellousML()
    print("Accuracy : ",Ret*100)


if __name__=="__main__":
    main()