import pre_processing
import dimensionality_reduction
import ml_algorithms
import evaluation
import dataframe 

import pandas as pd
import yfinance as yf

class Main:
    
    def __init__(self, stock_name, normalisation=False, savgol=False, PCA=0):
        
        ############################## Getting Historical data ##############################
        
        # self.file = pd.read_csv(file, sep='\t')
        ticker = yf.Ticker(stock_name).history(start="2020-01-01", end="2021-01-01") #retrieving all historical market data
        self.df = ticker[['Close']] # extracting closing market value for the stock
        # print(self.df)

        ############################## Populating dataframe ##############################

        df = dataframe.Dataframe(self.df) # creating 
        df.populate_data_frame()
        self.df = df.df
        # print(self.df)

        ############################## Preprocessing dataframe ##############################

        pp = pre_processing.Pre_processing(self.df)

        pp.clean_df()
        # print(pp.df)

        if normalisation:
            pp.normalisation()
        # print(pp.df)

        if savgol:
            pp.savitzky_golay(5, 2)
        # print(pp.df)

        self.df = pp.df
        # print(self.df)
        
        ############################## Dimensionality Reduction ##############################

        dr = dimensionality_reduction.Dimensionality_reduction(self.df)

        if PCA > 0:
            dr.pca(PCA)
            # dr.skpca(PCA)
        
        self.df = dr.df
        # print(self.df)
        
        ############################## Machine Learning ##############################

        ml_model = ml_algorithms.Ml_algorithms(self.df)
        ml_model.adj_df()
        # ml_model.svr()
        ml_model.knr()

        # #Testing
        # svr_model = Ml_algorithms.svr(file)
        # knn_model = Ml_algorithms.knn(file)

        # #evaluation
        # Evaluation.mae()

Main("MSFT", normalisation=True, savgol=False, PCA=10)