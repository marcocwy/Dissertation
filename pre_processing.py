import numpy as np
from scipy import signal as s

class Pre_processing:

    def __init__(self, df):
        self.df = df
        self.headings = list(self.df.columns.values)
        self.headings.pop(0) #removing first item in list, which is closing, as it does not need to be normalised

    def clean_df(self):
        print("Cleaning data frame...")
        contain_nan = []
        for i in range(self.df.shape[0]):
            if self.df.iloc[[i]].isnull().values.any():
                contain_nan += [i]
        # print(contain_nan)
        
        self.df.drop(self.df.index[contain_nan], inplace=True)
        # print(self.df)
        return self.df

    def normalisation(self): #put values between 0 and 1
        print("Normalising data frame...")
        for col in self.headings:
            self.df[col] = self.df[col] / self.df[col].sum()
        
        

    def savitzky_golay(self, n, order):  # n:odd, >=3, order <= n - 2
        print("Applying savitzky_golay filter to data frame...")
        for col in self.headings:
            vals = self.df[col].tolist()
            savgol = s.savgol_filter(vals, n, order).tolist()
            self.df[col] = savgol

