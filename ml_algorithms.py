import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
import pre_processing

class Ml_algorithms: 
    def __init__(self, df):
        self.df = df
        # self.label = []
        self.test = []

    def adj_df(self):
        self.test = self.df.iloc[-1].tolist()
        self.test.pop(0)

        self.df = self.df.shift(periods=1) 
        pp = pre_processing.Pre_processing(self.df)
        pp.clean_df()
        self.df = pp.df

    def get_data(self):
        df_data = self.df.drop('Close', axis=1)
        data = []
        for i in range(df_data.shape[0]):
            current = df_data.iloc[i].tolist()
            data += [current]

        # print(data)

        prices = self.df['Close'].tolist()
        # print(prices)
        return data, prices

    def svr(self):
        # print("Machine learning with SVR...")

        data, prices = self.get_data()

        # # lin_model  = SVR(kernel='linear', C=1e3)
        # # poly_model = SVR(kernel='poly', C=1e3, degree=2)
        rbf_model = SVR(kernel='rbf', C=1e3, gamma=0.1)

        rbf_model.fit(data, prices)

        predicted = rbf_model.predict([self.test])[0]

        # print("Predicted value: " + str(predicted))
        # print("Actual value: " + str(self.target))
        # per_error = abs(self.target-predicted)/predicted * 100
        # print("Percentage Error: " + str(per_error) + "%")

        return predicted

    def knr(self, n=3):

        # print("Machine learning with KNR...")

        data, prices = self.get_data()
        
        model = KNR(n)

        model.fit(data, prices)
        # print(self.test)
        predicted = model.predict([self.test])

        print("Predicted value: $" + str(predicted[0]))
        print("Actual value: $" + str(self.target))
        per_error = abs(self.target-predicted[0])/predicted[0] * 100
        print("Percentage Error: " + str(per_error) + "%")
