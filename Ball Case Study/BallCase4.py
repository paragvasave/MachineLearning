from sklearn import tree

def Ball_Predict(weight,surface):

    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    Labels =[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    obj = tree.DecisionTreeClassifier()
    obj = obj.fit(Features,Labels)
    
    ret = obj.predict([[weight,surface]])

    if(ret == 1):
        print("The object looks like Tennis ball.")
    else:
        print("The object looks like Cricket ball")


def main():

    print("Enter the weight of ball : ",end=" ")
    weight = int(input())

    print("Enter the surface of the ball [Rough/Smooth]: ",end=" ")
    surface = input()

    if surface.lower()=="rough":
        surface = 1
    elif surface.lower()=="smooth":
        surface = 0
    else:
        print("Invalid surface.")
        exit()
    
    Ball_Predict(weight,surface)

if __name__=="__main__":
    main()