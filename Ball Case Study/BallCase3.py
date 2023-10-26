# With Defined functions

from sklearn import tree 
# Encoding
# Rough - 1
# Smooth - 0
# Cricket - 2
# Tennis - 1


def Ball_Predict():
    
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    Labels =[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    # Decide the ML Alogorithm
    obj = tree.DecisionTreeClassifier()

    # Perform the training of model
    obj=obj.fit(Features,Labels)

    # Perform testing
    print(obj.predict([[97,0],[35,1]]))
    

def main():
    print("-------------- Python Machine Learnig ---------------")
    print("--------------Ball Predictor Case Study--------------")

    Ball_Predict()
    
if __name__=="__main__":
    main()