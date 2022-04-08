import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

class Ml_algorithms: 
    def __init__(self, df):
        self.df = df
        self.label = []
    
    def svr(self):
        print("Machine learning with SVR...")
        dates = self.df.index
        dates = dates.strftime('%Y%m%d').tolist()
        dates = np.reshape(dates,(len(dates), 1))
        # print(dates)
        x = [20020101]
        x = np.reshape(x,(len(x), 1))
        print(x)
        prices = self.df['Close'].tolist()
        # print(prices)
        # svr_lin  = SVR(kernel='linear', C=1e3)
        # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

        # svr_rbf.fit(dates, prices)
        # print(svr_rbf.predict([[20020101]]))
        # print(svr_rbf.predict([[1]]))
        # print(svr_rbf.predict([[30]]))


        # return null

    def knn(n=3):
        model = KNeighborsClassifier(n)

        # Train the model using the training sets
        model.fit(self.data, self.label)

        #Predict Output
        predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild

        return predicted