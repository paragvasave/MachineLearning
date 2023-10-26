import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def PlayPredictorModel(Data_File):
    Dataset = pd.read_csv(Data_File,index_col=0)

    features_names = ['Whether','Temperature']
    Whether = Dataset.Whether
    Temperature = Dataset.Temperature
    Play = Dataset.Play

    Encoder = preprocessing.LabelEncoder()
    Whether_Encoded = Encoder.fit_transform(Whether)
    Temperature_Encoded = Encoder.fit_transform(Temperature)
    Label = Encoder.fit_transform(Play)

    features = list(zip(Whether_Encoded,Temperature_Encoded))

    model = KNeighborsClassifier()
    model.fit(features,Label)

    Prediction = model.predict([[0,2]])
    if(Prediction==0):
        print("Dont Play")
    else:
        print("Play")


def main():
    PlayPredictorModel("PlayPredictor.csv")

if __name__=="__main__":
    main()