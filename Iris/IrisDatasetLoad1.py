# Loading iris dataset from load_iris and saving it in csv file. 
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pandas as pd

def CreateFile(FileName):

    iris = load_iris()
    iris = pd.DataFrame(data = np.c_[iris['data'],iris['target']],columns=iris['feature_names'] + ['target'])
    
    species =[]
    for i in range(len(iris['target'])):
        if iris['target'][i]==0:
            species.append('setosa')
        elif iris['target'][i]==1:
            species.append('versicolor')
        else:
            species.append('virginica')

    iris['species']=species
    iris = iris.to_csv()

    fd = open(FileName,'w',newline="")
    Data = (iris)
    fd.write(Data)
    fd.close()


def main():
    print("Enter the name of the file : ",end = " ")
    Name = input()
    CreateFile(Name)

if __name__=="__main__":
    main()



