import pre_processing
import dimensionality_reduction
import ml_algorithms
import evaluation
import dataframe 

import pandas as pd
import yfinance as yf

class Main:
    
    def __init__(self, stock_name, normalisation=False, savgol=False):
        # self.file = pd.read_csv(file, sep='\t')

        ticker = yf.Ticker(stock_name).history(period="max") #retrieving all historical market data
        closing = ticker[['Close']] # extracting closing market value for the stock

        df = dataframe.Dataframe(closing) # creating 
        df.populate_data_frame()
        stock = df.df
        # print(msft)

        # #preprocess file
        pp = pre_processing.Pre_processing(stock)

        pp.clean_df()
        # print(pp.df)

        if normalisation:
            pp.normalisation()
        # print(pp.df)

        if savgol:
            pp.savitzky_golay(5, 2)
        print(pp.df)
        
        # #dimensionality reduction
        # dr = dimensionality_reduction.Dimensionality_reduction(msft)
        # pca = dr.pca(10)
        # kpca = Dimensionality_reduction.kpca(x, 10)
        
        # #Training
        # svr_model = Ml_algorithms.svr(file)
        # knn_model = Ml_algorithms.knn(file)

        # #Testing
        # svr_model = Ml_algorithms.svr(file)
        # knn_model = Ml_algorithms.knn(file)

        # #evaluation
        # Evaluation.mae()

Main("MSFT", normalisation=False, savgol=True)