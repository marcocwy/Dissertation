import numpy as np
import pandas as pd

class Dimensionality_reduction: 

    def __init__(self, df):
        self.df = df

    def pca(self, n):
        
        data_set = self.df.drop(columns=['Close'])

        target = self.df['Close']

        # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        # data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
        
        # #prepare the data
        # data_set = data.iloc[:,0:4] #pandas frame
        # print(type(data_set))

        # #prepare the target
        # target = data.iloc[:,4]
        # print(type(target)) #pandas series

        #centering
        data_set_centered = data_set - np.mean(data_set, axis = 0) # mean:0, devi:1
        
        #calculating covariance matrix, eigen matrix and eigen vectors
        cov_matrix = np.cov(data_set_centered, rowvar = False)
        # print(cov_matrix)
        # eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        
        # #sorting eigenvalues in descending order
        # sorted_index = np.argsort(eigen_values)[::-1]
        # sorted_eigenvalue = eigen_values[sorted_index]
        # sorted_eigenvectors = eigen_vectors[:,sorted_index]
        
        # #selecting the first n components
        # n_eigenvectors = sorted_eigenvectors[:,0:n]
        
        # #transforming data to new labels
        # data_set_transformed = np.dot(n_eigenvectors.transpose(), data_set_centered.transpose()).transpose()
        
        # print(data_set_transformed)
        # # # return data_set_transformed

    def kpca(data_set, n):
        data_set_transformed = data_set
        return data_set_transformed