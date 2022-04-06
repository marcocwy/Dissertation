import pre_processing
import dimensionality_reduction
import ml_algorithms
import evaluation
import dataframe 

import pandas as pd
import yfinance as yf

class Main:
    
    def __init__(self):
        # self.file = pd.read_csv(file, sep='\t')

        ticker = yf.Ticker("MSFT").history(period="max") #retrieving all historical market data
        closing = ticker[['Close']] # extracting closing market value for the stock
        data = dataframe.Dataframe(closing) # creating 
        msft = data.populate_data_frame()
        # print(msft)

        # #preprocess file
        pp = pre_processing.Pre_processing(msft)
        # normalised = pp.normalisation()
        # print(normalised)
        savitzky_golay = pp.savitzky_golay(5, 2)
        print(savitzky_golay)
        
        # #dimensionality reduction
        # pca = Dimensionality_reduction.pca(x, 10)
        # kpca = Dimensionality_reduction.kpca(x, 10)
        
        # #Training
        # svr_model = Ml_algorithms.svr(file)
        # knn_model = Ml_algorithms.knn(file)

        # #Testing
        # svr_model = Ml_algorithms.svr(file)
        # knn_model = Ml_algorithms.knn(file)

        # #evaluation
        # Evaluation.mae()

Main()