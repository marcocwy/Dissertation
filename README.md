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
    -   LVF (Low Variance Filter)
-   Machine learning algorithms: 
    -   SVR (Support Vector Regression)                      change code
    -   KNN (K Nearest Neighbour)                            change code
-   Evaluation techniques:
    -   MAE (Mean Average Error)
    -   RMSE (Root Mean Square Error)
    -   R (R-Squared)

Features currently working on:


Features yet to implement:
-   Alter Main file to group functions better
-   Commenting code / functions

Dissertation:
- Abstract
    -   Rework
    -   Need a clean objective
-   Introduction
    -   
-   Literature Review
    -   Implement 3.4-3.6 into literature review
    -   Technical indicators
    -   What methods are currently used in the financial world
    -   What other dimensionality reduction methods are popular
    -   Combination explosion when clustering is used / regression
    -   Add research of Low variance filter
    -   Need to be able to justify why this dissertation is important/unique
    -   Concepts explained when necessary
    -   Comparison to similar attempts
    -   summary of the methods to date
    -   Check standard book chapter citation
-   Requirement and Analysis
    -   Explain each part with more technical details
    -   Technical indicators
    -   Use algorithms to explain the maths to show understanding
-   Plan
    -   Redo Gantt chart
-   Discussion of Code
    -   Method used
    -   Brief discussion of code
    -   Libraries used
        -   Include how scikit-learn is a well-establised library
-   Results
    -   Describe findings
    -   Unaltered database as baseline for others to comapre to
    -   Use graphs and tables
-   Evaluation
    -   How well the objectives were satisfied
    -   How appropriate the processed turned out to be
    -   Further directions of study 
    -   Relationships with other people's work