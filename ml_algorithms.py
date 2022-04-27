import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import pre_processing

class Ml_algorithms: 
    def __init__(self, df):
        self.df = df
        # self.label = []
        self.test = []

    def adj_df(self):
        '''
        correcting database, so that i technical indicators are generated from the close price of i-1 day
        ---------- Params ---------- 
        none  
        ---------- Returns ---------- 
        void, function transforms self.df
        '''
        self.test = self.df.iloc[-1].tolist()
        self.test.pop(0)

        self.df = self.df.shift(periods=1) 
        pp = pre_processing.Pre_processing(self.df)
        pp.clean_df()
        self.df = pp.df

    def get_data(self):
        '''
        return a list of data, and the target price
        '''
        df_data = self.df.drop('Close', axis=1)
        data = []
        for i in range(df_data.shape[0]):
            current = df_data.iloc[i].tolist()
            data += [current]

        # print(data)

        prices = self.df['Close'].tolist()
        
        return data, prices


    def detrending(self, data, prices):
        '''
        detrending by differencing, return a list of data, and the target price
        '''
        latest = prices[-1]
        prices = np.diff(prices)
        data.pop(0)
        return data, prices, latest


    def svr(self):
        '''
        Support vector machine regression 
        '''
        # print("Machine learning with SVR...")
    
        data, prices = self.get_data()

        data, prices, last_price = self.detrending(data, prices)
        
        rbf_model = SVR(kernel='rbf', C=1e3, gamma=0.1)

        rbf_model.fit(data, prices)
        
        # # plotting model
        # trend = rbf_model.predict(data)
        # # plot trend
        # pyplot.plot(prices)
        # pyplot.plot(trend)
        # pyplot.show()

        predicted = rbf_model.predict([self.test])[0]
        adjusted = last_price + predicted
        return adjusted
  

    def knr(self, n=3):
        '''
        K nearest neighbour regression
        '''

        # print("Machine learning with KNR...")

        data, prices = self.get_data()
        
        model = KNR(n)

        model.fit(data, prices)
        # print(self.test)
        predicted = model.predict([self.test])[0]

        # print("Predicted value: $" + str(predicted[0]))
        # print("Actual value: $" + str(self.target))
        # per_error = abs(self.target-predicted[0])/predicted[0] * 100
        # print("Percentage Error: " + str(per_error) + "%")

        return predicted
