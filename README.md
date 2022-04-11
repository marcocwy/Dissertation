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
    -   EMA (Exponential Moving Average)                     Library used
    -   MACD (Moving Average Convergence Divergence)         Library used (EMA)
    -   RSI (Relative Strength Index)                        Library used (SMA, EMA)
    -   PPO (Percentage Price Oscillator)                    Library used (EMA)
    -   SD (Standard Deviation)
    -   BIAS (Deviation rate)                                Library used (SMA, EMA)
    -   ROC (Rate of Change)
    -   K (Stochastic Oscillator %K)
    -   D (Stochastic Oscillator %D)
-   Preprocessing techniques:
    -   Cleaning database
    -   Normalisation
    -   Savitzky Golay smoothing                             Library used (Scipy)
-   Dimensionality reduction technique:
    -   PCA (Principal Component Analysis)                   change code
    -   KPCA (Kernel Principal Component Analysis)           change code
-   Machine learning algorithms: 
    -   SVR (Support Vector Regression)                      change code
    -   KNN (K Nearest Neighbour)                            change code
-   Evaluation techniques:
    -   MAE (Mean Average Error)
    -   RMSE (Root Mean Square Error)
    -   R (R-Squared)

Features currently working on:


Features yet to implement:
-   Machine learning algorithms:
    -   More potentially?
-   Commenting code / functions
-   Dissertation write up
