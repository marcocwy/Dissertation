import yfinance as yf
import numpy as np

class Data: 
    def __init__(self, stock):
        data = yf.Ticker(stock)
        hist = data.history(period="max") # get historical market data
        
        self.df = hist[['Close']] # Extract close

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
        # self.sma(5)
        # self.sma(6)
        # self.sma(10)
        # self.sma(20)
        # self.ema(5)
        # self.ema(6)
        # self.ema(10)
        # self.ema(20)
        # self.macd()
        # self.rsi(14, SMA=True)
        # self.rsi(14, SMA=False)
        # self.ppo()
        # self.sd(5)
        # self.bias(5, SMA=True)
        # self.bias(5, SMA=False)
        # self.roc(3)

        print(self.df)

    def sma(self, n):
        ''' 
        simple moving average with window size n
        '''
        heading = str(n) + "SMA" # column heading
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

        self.add_column(heading, vals)

    
    def ema(self, n):
        heading = str(n) +'EMA'
        vals = self.df.ewm(span = n, adjust=True, min_periods = n).mean() #add min_period
        vals = vals['Close'].tolist()
        self.add_column(heading, vals)

    def macd(self):
        '''
        Moving Average Convergence Divergence between 26 days and 12 days
        '''
        heading = 'MACD'
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

        self.add_column(heading, vals)
    
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
            heading = 'RSI(SMA)'
            ma_up = up.rolling(n).mean()
            ma_down = down.rolling(n).mean()
        else: #EMA
            heading = 'RSI(EMA)'
            ma_up = up.ewm(span = n, adjust=True, min_periods = n).mean()
            ma_down = down.ewm(span = n, adjust=True, min_periods = n).mean()

        up_list = ma_up.tolist()
        down_list = ma_down.tolist()

        # print(up_list)
        # print(down_list)
        vals = []
        for up, down in zip(up_list, down_list): # for every value in list
            rs = up / down #calculate relative strength
            rsi = 100 - (100/(1 + rs)) #calculate relative strength index
            vals.append(rsi)

        self.add_column(heading, vals)

    def ppo(self):
        '''
        Percentage Price Oscillator
        '''
        heading = 'PPO'
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

        self.add_column(heading, vals)


    def sd(self, n):
        ''' 
        standard deviation indicator with window size n
        '''
        heading = str(n) + "SD" # column heading
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
        self.add_column(heading, vals)

    def bias(self, n, SMA):
        '''
        Deviation rate (BIAS) indicator with window size n
        '''

        if SMA:
            heading = 'BIAS(SMA)'
            ma = self.df['Close'].rolling(n).mean()
        else: #EMA
            heading = 'BIAS(EMA)'
            ma = self.df['Close'].ewm(span = n, adjust=True, min_periods = n).mean()

        ma_list = ma.tolist()

        vals = []
        for ma, val in zip(ma_list, self.df['Close']): # for every value in list
            dev_rate = (val - ma) / ma * 100
            vals.append(dev_rate)

        self.add_column(heading, vals)

    def roc(self, n):
        '''
        Rate of Change indicator comparing current price and n days ago
        '''
        heading = str(n) + "ROC" # column heading
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
        print(vals)

        self.add_column(heading, vals)

                
