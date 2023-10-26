
import numpy as np
 
def LogisticRegression(X):
    
    # Y=1/(1+e^-X)
    eulers = np.e
    result=np.power(eulers,-X)
    Y = 1/(1+result)
    print(Y)

    if(Y>0.5):
        print('Survived')
    else:
        print('Did not survived')

def main():
    print("------------Logistic Regression Userdefined------------")
    print("Enter value of x")
    Value = int(input())

    LogisticRegression(Value)

if __name__=="__main__":
    main()
