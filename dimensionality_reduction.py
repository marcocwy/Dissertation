import numpy as np
import pandas as pd

class Dimensionality_reduction: 

    def __init__(self, df):
        self.df = df

    def pca(self, n):
        print("Applying PCA to data frame...")
        df = self.df.drop(columns=['Close'])

        target = pd.DataFrame(self.df['Close'])


        #centering
        df_centered = df - np.mean(df, axis = 0) # mean:0, devi:1
        
        #calculating covariance matrix, eigen matrix and eigen vectors
        cov_mat = np.cov(df_centered, rowvar = False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
        
        #sorting eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        
        #selecting the first n components
        n_eigenvectors = sorted_eigenvectors[:,0:n]
        
        # #transforming data to new labels
        data_set_transformed = np.dot(n_eigenvectors.transpose(), df_centered.transpose()).transpose()
        
        # print(data_set_transformed)
        headings = []
        for i in range(n):
            heading = "PC" + str(i+1)
            headings += [heading]
        # print(headings)
        pc_vals = pd.DataFrame(data_set_transformed, index = target.index ,columns = headings)
        # print(principal_df)

        pc_df = pd.concat([target, pc_vals] , axis = 1)
        # print(pc_df)

        self.df = pc_df

    def kpca(data_set, n):
        data_set_transformed = data_set
        return data_set_transformed