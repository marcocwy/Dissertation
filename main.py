import pre_processing
import dimensionality_reduction
import ml_algorithms
import evaluation
import features 

import pandas as pd
import yfinance as yf
import datetime as dt

class Main:
    
    def __init__(self, stock_name, savgol=False, PCA=0, KPCA=0, LVF=0, SVR=True, KNR=False, T=5):
        
        ############################## Get historical data ##############################
        start = "2020-03-02"
        end = "2021-03-03"

        self.df , self.pred = self.get_historical_data(stock_name, start, end, T)

        print('Predicting next '+str(T)+ ' days of '+stock_name+' market close price...')
        pred_vals = self.predict_period(savgol, PCA, KPCA, LVF, SVR, KNR, T)

        # self.show_results(pred_vals)

        
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

    def predict(self, dframe, savgol, PCA, KPCA, LVF, SVR, KNR):
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

        if PCA > 0 or KPCA > 0 or LVF > 0:
            pp.normalisation()

        if savgol:
            pp.savitzky_golay(5, 2)
        # print(pp.df)

        df = pp.df
        print(df)
        
        ############################## Dimensionality Reduction ##############################

        dr = dimensionality_reduction.Dimensionality_reduction(df)

        if PCA > 0:
            dr.pca(PCA)
            # dr.skpca(PCA)
        
        if KPCA > 0:
            dr.kpca(KPCA)
        
        if LVF > 0:
            dr.lvf(LVF)

        df = dr.df
        # print(df)
        
        ############################## Machine Learning ##############################

        ml_model = ml_algorithms.Ml_algorithms(df)
        ml_model.adj_df()

        if SVR:
            predicted = ml_model.svr()

        if KNR:
            predicted = ml_model.knr() 

        return predicted

    def predict_period(self, savgol, PCA, KPCA, LVF, SVR, KNR, T):
        
        predicted = self.predict(self.df, savgol, PCA, KPCA, LVF, SVR, KNR)

        pred_vals = [(predicted, self.pred['Close'][0])]

        temp_df = self.pred.copy(deep=True)
        new_df = self.df.copy(deep=True)

        for i in range(T-1):
            temp_df = self.pred.iloc[i].to_frame().transpose()
            temp_df = temp_df.assign(Close=predicted)
            new_df = new_df.append(temp_df)
            predicted = self.predict(new_df, savgol, PCA, KPCA, LVF, SVR, KNR)
            pred_vals += [(predicted, self.pred['Close'][i+1])]
        
        # print(pred_vals)

        return pred_vals

    def show_results(self, pred):

        pred_df = self.pred.copy(deep=True)
        pred_df = pred_df.drop(columns=['Close'])
        pred_close = [x[0] for x in pred]
        actual_close = [x[1] for x in pred]
        pred_df['Predicted Close'] = pred_close
        pred_df['Actual Close'] = actual_close
        
        print(pred_df)

        e = evaluation.Evaluation(pred)
        mae = e.mae()
        rmse = e.rmse()
        r = e.r()

        print("MAE:" + str(mae))
        print("RMSE:" + str(rmse))
        print("R-Squared:" + str(r))

Main("MSFT", savgol=False, KPCA=10, T=1)