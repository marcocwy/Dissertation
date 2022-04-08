import numpy as np
import pandas as pd

class Dataframe: 
    def __init__(self, df):        
        self.df = df
        pd.options.mode.chained_assignment = None # default='warn', turn off chain assignment warning

    def add_column(self, heading, values):
        '''
        add calculated values to data frame
        '''
        self.df[heading] = values

    def populate_data_frame(self):
        '''
        Poplate data frame with features
        '''
        # self.df['MA'] = self.df.rolling(window=5).mean()
        print('Populating data frame...')
        self.add_column("5SMA"        ,self.sma(5))
        self.add_column("6SMA"        ,self.sma(6))
        self.add_column("10SMA"       ,self.sma(10))
        self.add_column("20SMA"       ,self.sma(20))
        self.add_column("5EMA"        ,self.ema(5))
        self.add_column("6EMA"        ,self.ema(6))
        self.add_column("10EMA"       ,self.ema(10))
        self.add_column("20EMA"       ,self.ema(20))
        self.add_column("MACD"        ,self.macd()) 
        self.add_column("14RSI(SMA)"  ,self.rsi(14, SMA=True)) 
        self.add_column("14RSI(EMA)"  ,self.rsi(14, SMA=False)) 
        self.add_column("PPO"         ,self.ppo()) 
        self.add_column("5SD"         ,self.sd(5))
        self.add_column("BIAS(5SMA)"  ,self.bias(5, SMA=True))
        self.add_column("BIAS(5EMA)"  ,self.bias(5, SMA=False))
        self.add_column("BIAS(10SMA)" ,self.bias(10, SMA=True))
        self.add_column("BIAS(10EMA)" ,self.bias(10, SMA=False))
        self.add_column("5ROC"        ,self.roc(5))
        self.add_column("6ROC"        ,self.roc(6))
        self.add_column("10ROC"       ,self.roc(10))
        self.add_column("20ROC"       ,self.roc(20))
        self.add_column("9K"          ,self.sok(9))
        self.add_column("9D"          ,self.sod(9, 3))
        

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

        return vals

    
    def ema(self, n):

        vals = self.df.ewm(span = n, adjust=True, min_periods = n).mean() #add min_period
        vals = vals['Close'].tolist()

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
