# Loading iris dataset from URL and saving it in csv file. 
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pandas as pd

def CreateFile(FileName):

    url = 'https://drive.google.com/file/d/1wD-TBL1rtL2WMLHMTCJY9eRcD-SqqESA/view?usp=drive_link'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    iris = pd.read_csv(url)

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
