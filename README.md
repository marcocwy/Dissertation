# Dissertation
Maching Learning with high dimensional data with stocks data

python main.py # to run script
Toggle features in Main():
    stock_name       :String        # Name of stock
    normalisation    :Boolean       # default=False
    savgol           :Boolean       # default=False
    PCA              :INT > 0       # default=0 (Not activated), Positive number of principle components

Features implemented:
-   Collect data from yFinance
-   Populate database with indices:
    -   SMA (Simple Moving Average)
    -   EMA (Exponential Moving Average)
    -   MACD (Moving Average Convergence Divergence)
    -   RSI (Relative Strength Index)
    -   PPO (Percentage Price Oscillator)
    -   SD (Standard Deviation)
    -   BIAS (Deviation rate)
    -   ROC (Rate of Change)
    -   K (Stochastic Oscillator %K)
    -   D (Stochastic Oscillator %D)
-   Preprocessing techniques:
    -   Cleaning database
    -   Normalisation
    -   Savitzky Golay smoothing 
-   Dimensionality reduction technique:
    -   PCA (Principal Component Analysis)

Features currently working on:
-   Machine learning algorithms: 
    # Both currently working, need to implement so that both will do rolling windows
    -   SVR (Support Vector Regression)
    -   KNN (K Nearest Neighbour)

Features yet to implement:
-   Dimensionality reduction techniques:
    -   KPCA (Kernel Principal Component Analysis)
-   Machine learning algorithms:
    -   More potentially?
-   Evaluation techniques:
    -   MAE (Mean Average Error)
    -   RMSE (Root Mean Square Error)
    -   R (R-Squared)
