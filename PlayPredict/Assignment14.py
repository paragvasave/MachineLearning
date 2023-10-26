from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def PlayPredictorModel(Data):
    Dataset = pd.read_csv(Data, index_col=0)
    features_names = ["Whether","Temperature"]

    Whether = Dataset.Whether
    Temperature = Dataset.Temperature
    Play = Dataset.Play

    Encoder = preprocessing.LabelEncoder()

    whether_encoded = Encoder.fit_transform(Whether)
    temp_encoded = Encoder.fit_transform(Temperature)
    Label = Encoder.fit_transform(Play)

    features= list(zip(whether_encoded,temp_encoded))

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(features,Label)

    Prediction = model.predict([[0,2]])
    print(Prediction)


def main():
    PlayPredictorModel("PlayPredictor.csv")

if __name__=="__main__":
    main()