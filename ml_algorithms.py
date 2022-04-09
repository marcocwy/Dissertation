import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import pre_processing

class Ml_algorithms: 
    def __init__(self, df):
        self.df = df
        # self.label = []
        self.test = []
        self.target = 0

    def adj_df(self):
        self.test = self.df.iloc[-1].tolist()
        self.target = self.test[0]
        self.test.pop(0)

        self.df = self.df.shift(periods=1) 
        pp = pre_processing.Pre_processing(self.df)
        pp.clean_df()
        self.df = pp.df

    def svr(self):
        print("Machine learning with SVR...")
        # dates = self.df.index
        # dates = dates.strftime('%Y%m%d').tolist()
        # dates = np.reshape(dates,(len(dates), 1))
        # print(dates)
        df_data = self.df.drop('Close', axis=1)
        data = []
        for i in range(df_data.shape[0]):
            current = df_data.iloc[i].tolist()
            data += [current]

        # print(data)

        # x = [20020101]
        # x = np.reshape(x,(len(x), 1))
        # print(x)
        prices = self.df['Close'].tolist()
        # print(prices)
        # # svr_lin  = SVR(kernel='linear', C=1e3)
        # # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

        svr_rbf.fit(data, prices)
        # print(self.test)
        predicted = svr_rbf.predict([self.test])

        print("Predicted value: " + str(predicted[0]))
        print("Actual value: " + str(self.target))


        # return null

    def knn(n=3):
        model = KNeighborsClassifier(n)

        # Train the model using the training sets
        # model.fit(self.data, self.label)

        #Predict Output
        predicted = model.predict([[0,2]]) # 0:Overcast, 2:Mild

        return predicted