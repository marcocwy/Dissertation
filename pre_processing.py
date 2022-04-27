import numpy as np
from scipy import signal as s

class Pre_processing:

    def __init__(self, df):
        self.df = df

    def clean_df(self):
        '''
        removing any records containing nan value
        '''
        # print("Cleaning data frame...")
        contain_nan = []
        for i in range(self.df.shape[0]):
            if self.df.iloc[[i]].isnull().values.any():
                contain_nan += [i]
        # print(contain_nan)
        
        self.df.drop(self.df.index[contain_nan], inplace=True)
        # print(self.df)
        return self.df

