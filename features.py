import numpy as np
import pandas as pd
from scipy import signal as s
import matplotlib.pyplot as plt
import math

class Features: 
    def __init__(self, df, savgol):        
        self.df = df
        self.savgol = savgol
        pd.options.mode.chained_assignment = None # default='warn', turn off chain assignment warning

    def add_column(self, heading, values):
        '''
        add calculated values to data frame
        '''
        self.df[heading] = values

    def savitzky_golay(self, n, order):  # n:odd, >=3, order <= n - 2
        '''
        savitzky golay smoothing filter
        '''
        vals = self.df['Close'].tolist()
        savgol = s.savgol_filter(vals, n, order).tolist()
        return savgol

    def populate_data_frame(self):
        '''
        Poplate data frame with features
        '''
        close = self.df['Close'].tolist()
        if self.savgol:
            savgol = self.savitzky_golay(5, 2)
            # print(savgol)
            self.df['Close'] = savgol

        # self.df['MA'] = self.df.rolling(window=5).mean()
        # print('Populating data frame...')
        self.add_column("5SMA"        ,self.sma(5))
        self.add_column("6SMA"        ,self.sma(6))
        self.add_column("15SMA"       ,self.sma(15))
        self.add_column("25SMA"       ,self.sma(25))
        self.add_column("5EMA"        ,self.ema(5))
        self.add_column("6EMA"        ,self.ema(6))
        self.add_column("15EMA"        ,self.ema(15))
        self.add_column("25EMA"       ,self.ema(25))
        self.add_column("MACD"        ,self.macd()) 
        self.add_column("14RSI(SMA)"  ,self.rsi(14, SMA=True))
        self.add_column("25RSI(SMA)"  ,self.rsi(25, SMA=True))
        self.add_column("14RSI(EMA)"  ,self.rsi(14, SMA=False))
        self.add_column("25RSI(EMA)"  ,self.rsi(25, SMA=False))
        self.add_column("PPO"         ,self.ppo()) 
        self.add_column("5SD"         ,self.sd(5))
        self.add_column("6SD"         ,self.sd(6))
        self.add_column("15SD"         ,self.sd(15))
        self.add_column("25SD"         ,self.sd(25))
        self.add_column("BIAS(5SMA)"  ,self.bias(5, SMA=True))
        self.add_column("BIAS(6SMA)" ,self.bias(6, SMA=True))
        self.add_column("BIAS(15SMA)" ,self.bias(15, SMA=True))
        self.add_column("BIAS(25SMA)" ,self.bias(25, SMA=True))
        self.add_column("BIAS(5EMA)"  ,self.bias(5, SMA=False))
        self.add_column("BIAS(6EMA)" ,self.bias(6, SMA=False))
        self.add_column("BIAS(15EMA)"  ,self.bias(15, SMA=False))
        self.add_column("BIAS(25EMA)" ,self.bias(25, SMA=False))
        self.add_column("5ROC"        ,self.roc(5))
        self.add_column("6ROC"        ,self.roc(6))
        self.add_column("15ROC"       ,self.roc(15))
        self.add_column("25ROC"       ,self.roc(25))
        self.add_column("9K"          ,self.sok(9))
        self.add_column("9D"          ,self.sod(9, 3))

        if self.savgol:
            self.df['Close'] = close
        # print(self.df)

        self.df = self.df.dropna()

    def detrending(self, data):
        '''
        detrending by differencing ignoring the fist n Nan values, return a list of detrended data
        '''
        nans = []
        vals = []
        for val in data:
            if math.isnan(val):
                nans += [val]
            else:
                vals += [val]
        vals = np.diff(vals).tolist()
        data = nans + [np.nan] + vals
        return data

    def sma(self, n):
        ''' 
        return a list of simple moving average with window size n
        '''
        vals = [] # list of all moving average values

        for i in range(n - 1): # populating first n-1 items since values can not be calculated
            vals += [np.nan]

        length = self.df.shape[0] - (n - 1) # set length to remaining values to be calculated
        for i in range(length):
            current = i + (n - 1) # current value to be calculated which is n-1 values ahead
            val = 0

            for j in range(n): #sum of latest n values
                # print(j)
                val += self.df['Close'].values[current - j]

            val /= n #work out average
            vals += [val]

        vals = self.detrending(vals)
        return vals

    
    def ema(self, n):

        vals = self.df.ewm(span = n, adjust=True, min_periods = n).mean() #add min_period
        vals = vals['Close'].tolist()

        vals = self.detrending(vals)

        return vals

    def macd(self):
        '''
        Moving Average Convergence Divergence between 26 days and 12 days
        '''
        fast = 12
        slow = 26
        vals1 = self.df.ewm(span = fast, adjust=True, min_periods = fast).mean() #Fast length EMA of 12 days
        vals2 = self.df.ewm(span = slow, adjust=True, min_periods = slow).mean() #Slow length EMA of 26 days

        #convert to list
        vals1 = vals1['Close'].tolist()
        vals2 = vals2['Close'].tolist()
        
        vals = []
        for val1, val2 in zip(vals1, vals2): # for every value in list
            vals.append(val1 - val2) #calculate difference

        return vals
    
    def rsi(self, n, SMA):
        '''
        Relative Strength Index
        '''
        df_diff = self.df['Close'].diff() # difference between ith and (i-1)th 

        up = df_diff.clip(lower=0) #0 if -ive, leave if +ive
        down = abs(df_diff.clip(upper=0)) #0 if +ive, absolute value if -ve
        # print(down)

        #ema up
        if SMA:
            ma_up = up.rolling(n).mean()
            ma_down = down.rolling(n).mean()
        else: #EMA
            ma_up = up.ewm(span = n, adjust=True, min_periods = n).mean()
            ma_down = down.ewm(span = n, adjust=True, min_periods = n).mean()

        up_list = ma_up.tolist()
        down_list = ma_down.tolist()

        vals = []
        for up, down in zip(up_list, down_list): # for every value in list
            rs = up / down #calculate relative strength
            rsi = 100 - (100/(1 + rs)) #calculate relative strength index
            vals.append(rsi)

        return vals

    def ppo(self):
        '''
        Returns Percentage Price Oscillator
        '''
        fast = 12
        slow = 26
        vals1 = self.df.ewm(span = fast, adjust=True, min_periods = fast).mean() #Fast length EMA of 12 days
        vals2 = self.df.ewm(span = slow, adjust=True, min_periods = slow).mean() #Slow length EMA of 26 days
        
        #convert to list
        vals1 = vals1['Close'].tolist()
        vals2 = vals2['Close'].tolist()
        
        vals = []
        for val1, val2 in zip(vals1, vals2): # for every value in list
            diff = (val1 - val2) #calculate difference
            val = diff / val2 * 100
            vals.append(val)

        return vals


    def sd(self, n):
        ''' 
        standard deviation indicator with window size n
        '''
        vals = [] # list of all moving average values

        for i in range(n - 1): # populating first n-1 items since values can not be calculated
            vals += [np.nan]

        length = self.df.shape[0] - (n - 1) # set length to remaining values to be calculated
        for i in range(length):
            current = i + (n - 1) # current value to be calculated which is n-1 values ahead
            val_n = [] #list of latest n values

            for j in range(n): 
                val_n += [self.df['Close'].values[current - j]]
            
            val = np.std(val_n)
            vals += [val]
        return vals

    def bias(self, n, SMA):
        '''
        Deviation rate (BIAS) indicator with window size n
        '''

        if SMA:
            ma = self.df['Close'].rolling(n).mean()
        else: #EMA
            ma = self.df['Close'].ewm(span = n, adjust=True, min_periods = n).mean()

        ma_list = ma.tolist()

        vals = []
        for ma, val in zip(ma_list, self.df['Close']): # for every value in list
            dev_rate = (val - ma) / ma * 100
            vals.append(dev_rate)

        return vals

    def roc(self, n):
        '''
        Rate of Change indicator comparing current price and n days ago
        '''
        vals = []
        length = self.df.shape[0] # set length to remaining values to be calculated
        for i in range(length):
            if (i - n) < 0:
                roc = np.nan
            else:
                current_val = self.df['Close'].values[i]
                previous_val = self.df['Close'].values[i-n]
                roc = (current_val - previous_val) / previous_val * 100

            vals.append(roc)

        return vals

    def sok(self, n):
        '''
        Stochastic Oscillator %K with the period n
        '''
        min_vals = self.df.rolling(n).min() #min value within n days period
        max_vals = self.df.rolling(n).max() #max value within n days period

        vals = []
        length = self.df.shape[0]
        for i in range(length): # for every value in list
            min = min_vals['Close'].values[i]
            max = max_vals['Close'].values[i]
            cur = self.df['Close'].values[i]
            val = (cur - min) / (max - min) * 100
            vals.append(val)
    
        return vals
    
    def sod(self, kn, dn):
        '''
        Stochastic Oscillator %D with the smoothing period dn and span kn
        '''   
        k = self.sok(kn)
        vals_series = pd.Series(k)
        d = vals_series.rolling(dn).mean()
        vals = d.tolist()

        return vals
