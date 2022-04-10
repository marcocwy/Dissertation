import pre_processing
import dimensionality_reduction
import ml_algorithms
import evaluation
import features 

import pandas as pd
import yfinance as yf
import datetime as dt

class Main:
    
    def __init__(self, stock_name, normalisation=False, savgol=False, PCA=0, KPCA=0, SVR=True, KNR=False, T=5):
        
        ############################## Get historical data ##############################
        start = "2020-03-02"
        end = "2021-03-03"

        self.df , self.pred = self.get_historical_data(stock_name, start, end, T)
        # self.org = self.get_historical_data(stock_name, start, end, T)[0]

        # print(self.df)
        # print(self.pred)

        self.predict_period(normalisation, savgol, PCA, KPCA, SVR, KNR, T)


        
    def get_historical_data(self, stock_name, start, end, T):

        ticker = yf.Ticker(stock_name).history(start=start, end=end) #retrieving all historical market data (fisrt val 2020-03-02, last val 2021-03-02)
        df = ticker[['Close']] # extracting closing market value for the stock
        
        i = 0
        pred_end = end
        while True:
            yr = int(pred_end[:4])
            mm = int(pred_end[5:7])
            dd = int(pred_end[8:])
            pred_end = dt.datetime(yr,mm,dd) + dt.timedelta(days=1)
            pred_end = pred_end.strftime("%Y-%m-%d")

            pred = yf.Ticker(stock_name).history(start=end, end=pred_end)
            pred_len = pred.shape[0]
            
            if pred_len == T:
                break
        
        pred = pred[['Close']]
        # print(pred)

        return df, pred

    def predict(self, dframe, normalisation, savgol, PCA, KPCA, SVR, KNR):
        '''
        predict T+1 stock value
        '''
        df = dframe.copy(deep=True)
        ############################## Populating dataframe ##############################

        f = features.Features(df)
        f.populate_data_frame()   
        
        df = f.df

        ############################## Preprocessing dataframe ##############################

        pp = pre_processing.Pre_processing(df)

        pp.clean_df()
        # print(pp.df)

        if normalisation:
            pp.normalisation()
        # print(pp.df)

        if savgol:
            pp.savitzky_golay(5, 2)
        # print(pp.df)

        df = pp.df
        # print(self.df)
        
        ############################## Dimensionality Reduction ##############################

        dr = dimensionality_reduction.Dimensionality_reduction(df)

        if PCA > 0:
            dr.pca(PCA)
            # dr.skpca(PCA)
            # dr.kpca(PCA)
        
        df = dr.df
        # print(df)
        
        ############################## Machine Learning ##############################

        ml_model = ml_algorithms.Ml_algorithms(df)
        ml_model.adj_df()
        predicted = ml_model.svr()
        # predicted, target = ml_model.knr()

        return predicted

    def predict_period(self, normalisation, savgol, PCA, KPCA, SVR, KNR, T):
        
        predicted = self.predict(self.df, normalisation, savgol, PCA, KPCA, SVR, KNR)
        print (predicted, self.pred['Close'][0])
        # print(self.df)

        temp_df = self.pred.copy(deep=True)
        new_df = self.df.copy(deep=True)

        # temp_df = self.pred.iloc[0].to_frame().transpose()
        # temp_df = temp_df.assign(Close=predicted)
        # new_df = new_df.append(temp_df)
        # predicted, target = self.predict(normalisation, savgol, PCA, KPCA, SVR, KNR)
        # print (predicted, target)
        # print(new_df)
        for i in range(T-1):
            temp_df = self.pred.iloc[i].to_frame().transpose()
            temp_df = temp_df.assign(Close=predicted)
            new_df = new_df.append(temp_df)
            predicted = self.predict(new_df, normalisation, savgol, PCA, KPCA, SVR, KNR)
            print (predicted, self.pred['Close'][i+1])

        print(new_df) #last set of data not added
        print(self.pred)
        print(self.df)

Main("MSFT", normalisation=True, savgol=False, PCA=10, T=5)