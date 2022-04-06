import numpy as np
from scipy import signal as s

class Pre_processing:

    def __init__(self, df):
        self.df = df
        self.headings = list(self.df.columns.values)
        self.headings.pop(0) #removing first item in list, which is closing, as it does not need to be normalised
    def normalisation(self):

        for col in self.headings:
            self.df[col] = self.df[col] / self.df[col].sum()
        
        return self.df
        

    def savitzky_golay(self, n, order):  # n:odd, >=3, order <= n - 2

        for col in self.headings:
            vals = self.df[col].tolist()
            savgol = s.savgol_filter(vals, n, order).tolist()
            self.df[col] = savgol
        return self.df

