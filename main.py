import pre_processing
import dimensionality_reduction
import ml_algorithms
import evaluation
import data 

import pandas as pd
import yfinance as yf

class Main:
    
    def __init__(self):
        # self.file = pd.read_csv(file, sep='\t')

        ticker = yf.Ticker("MSFT").history(period="max")
        closing = ticker[['Close']]
        msft = data.Data(closing)
        msft.populate_data_frame()

        # #preprocess file
        # normalised = Pre_processing.normalisation(self.file)
        # savitzky_golay = Pre_processing.savitzky_golay(self.file)

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