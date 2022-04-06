import numpy as np
from scipy import signal as s

class Pre_processing:

    def __init__(self, df):
        self.df = df
        self.headings = list(self.df.columns.values)
        self.headings.pop(0) #removing first item in list, which is closing, as it does not need to be normalised

    def clean_df(self):
        cleaned = self.df
        # cleaned['Date'] = cleaned.index
        # cleaned.dropna(inplace=True)
        # cleaned.reset_index(drop=True, inplace=True)
        
        # # cleaned.set_index('Date',  inplace=True)
        # self.df = cleaned
        # print(self.df)

        # for row in cleaned.index:
            
        # return cleaned

    def normalisation(self): #put values between 0 and 1

        for col in self.headings:
            self.df[col] = self.df[col] / self.df[col].sum()
        
        return self.df
        

    def savitzky_golay(self, n, order):  # n:odd, >=3, order <= n - 2

        for col in self.headings:
            vals = self.df[col].tolist()
            savgol = s.savgol_filter(vals, n, order).tolist()
            self.df[col] = savgol
        return self.df

