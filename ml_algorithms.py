import numpy as np
import sklearn as sk

class Ml_algorithms: 
    def __init__(self, file):
        self.data = file
        self.label = []
    
    def svr(data):
        return null

    def knn(n=3):
        model = KNeighborsClassifier(n)

        # Train the model using the training sets
        model.fit(self.data, self.label)

        #Predict Output
        predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild

        return predicted