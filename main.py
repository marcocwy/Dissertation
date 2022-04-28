import pre_processing
import dimensionality_reduction
import ml_algorithms
import evaluation
import features 

import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

class Main:
    
    def __init__(self, stock_name, savgol=False, PCA=0, KPCA=0, LVF=0, SVR=False, KNR=False, T=10):
        
        ############################## Get historical data ##############################
        start = "2020-03-02"
        end = "2021-03-03"

        self.df , self.pred = self.get_historical_data(stock_name, start, end, T)

        self.print_configurations(stock_name, savgol, PCA, KPCA, LVF, SVR, KNR, T)
        pred_vals = self.predict_period(savgol, PCA, KPCA, LVF, SVR, KNR, T)

        self.show_results(pred_vals)

    def print_configurations(self, stock_name, savgol, PCA, KPCA, LVF, SVR, KNR, T):
        '''
        print configuration in console
        '''
        print('Predicting next '+str(T)+ ' days of '+stock_name+' market close price...')

        if savgol:
            savgol = "On"
        else:
            savgol = "Off"
        
        dr_method = "None"
        if PCA > 0:
            dr_method = "PCA"

        if KPCA > 0:
            dr_method = "KPCA"

        if LVF > 0:
            dr_method = "LVF"

        if SVR:
            ml_method = "SVR"

        if KNR:
            ml_method = "KNR"
        
        print('SavGol: '+savgol+ ', dimensional reduction method: '+dr_method+', Machine learning algorithm: '+ ml_method)
        
    def get_historical_data(self, stock_name, start, end, T):
        '''
        Return data given the stock and period of data, as well as test data
        '''
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
        predict stock price of the next day
        '''
        df = dframe.copy(deep=True)
        ############################## Populating dataframe ##############################

        f = features.Features(df, savgol)
        f.populate_data_frame()   
        
        df = f.df.copy(deep=True)

        ############################## Preprocessing dataframe ##############################

        pp = pre_processing.Pre_processing(df)
        pp.clean_df()
        # print(pp.df)
        df = pp.df.copy(deep=True)
        # print(df)
        
        ############################## Dimensionality Reduction ##############################

        dr = dimensionality_reduction.Dimensionality_reduction(df)

        if PCA > 0:
            dr.pca(PCA)
            # dr.skpca(PCA)
        
        if KPCA > 0:
            dr.kpca(KPCA)
        
        if LVF > 0:
            dr.lvf(LVF)

        df = dr.df.copy(deep=True)
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
        '''
        Rolling window of prediction for the next T days 
        '''
        predicted = self.predict(self.df, savgol, PCA, KPCA, LVF, SVR, KNR)
        # print(predicted)
        pred_vals = [(predicted, self.pred['Close'][0])]

        temp_df = self.pred.copy(deep=True)
        new_df = self.df.copy(deep=True)

        for i in range(T-1):
            temp_df = self.pred.iloc[i].to_frame().transpose()
            temp_df = temp_df.assign(Close=predicted)
            new_df = pd.concat([new_df, temp_df])
            # new_df = new_df.append(temp_df)
            # print(new_df)
            # print(new_df2)
            predicted = self.predict(new_df, savgol, PCA, KPCA, LVF, SVR, KNR)
            pred_vals += [(predicted, self.pred['Close'][i+1])]
            # print(predicted)
        
        # print(pred_vals)

        return pred_vals

    def show_results(self, pred):
        '''
        print evaluation metrics in console
        '''
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

        temp_df = self.df.copy(deep=True)
        temp_df.rename(columns={'Close': 'Actual Close'}, inplace=True)
        # print(temp_df)
        new_df = pd.concat([temp_df, pred_df])

        # new_df.plot(kind = 'line')
        # plt.show()
        
STOCK = "NVDA"
# Main(STOCK, savgol=False, SVR=True, T=10)
# Main(STOCK, savgol=True, SVR=True, T=10)
# Main(STOCK, savgol=False, PCA=10, SVR=True, T=10)
# Main(STOCK, savgol=True, PCA=10, SVR=True, T=10)
# Main(STOCK, savgol=False, KPCA=10, SVR=True, T=10)
# Main(STOCK, savgol=True, KPCA=10, SVR=True, T=10)
# Main(STOCK, savgol=False, LVF=10, SVR=True, T=10)
# Main(STOCK, savgol=True, LVF=10, SVR=True, T=10)

Main(STOCK, savgol=False, KNR=True, T=10)
Main(STOCK, savgol=True, KNR=True, T=10)
Main(STOCK, savgol=False, PCA=10, KNR=True, T=10)
Main(STOCK, savgol=True, PCA=10, KNR=True, T=10)
Main(STOCK, savgol=False, KPCA=10, KNR=True, T=10)
Main(STOCK, savgol=True, KPCA=10, KNR=True, T=10)
Main(STOCK, savgol=False, LVF=10, KNR=True, T=10)
Main(STOCK, savgol=True, LVF=10, KNR=True, T=10)