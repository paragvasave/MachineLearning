import pandas as pd
from sklearn.cluster import KMeans

def Cluster():
    dataset=pd.read_csv("Data.csv")
    x=dataset.iloc[:,[1,2,3]].values
    
    wcss=[]

    for i in range(1,11):
        kmeans =KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    print(pd.DataFrame(wcss))





def main():
    Cluster()

if __name__=="__main__":
    main()