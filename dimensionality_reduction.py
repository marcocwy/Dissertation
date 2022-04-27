import numpy as np
import pandas as pd
# from sklearn.decomposition import PCA

import scipy as sp

class Dimensionality_reduction: 

    def __init__(self, df):
        self.df = df

    def normalisation(self):
        '''
        Normalisation
        '''
        headings = list(self.df.columns.values)
        headings.pop(0) #removing first item in list, which is closing, as it does not need to be normalised
        for col in headings:
            self.df[col] = self.df[col] / self.df[col].sum()


    def pca(self, n):
        '''
        Principle component analysis to reduce dimensions
        ---------- Params ---------- 
        n: int
            Number of principal components to return    
        ---------- Returns ---------- 
        void, function transforms self.df
        '''
        # print("Applying PCA to data frame...")
        self.normalisation()
        df = self.df.drop(columns=['Close'])

        target = pd.DataFrame(self.df['Close'])

        
        #centering
        df_centered = df - np.mean(df, axis = 0) # mean:0, devi:1
        
        #calculating covariance matrix, eigen matrix and eigen vectors
        cov_mat = np.cov(df_centered, rowvar = False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        
        #sorting eigenvalues in descending order
        sorted_i = np.argsort(eig_vals)[::-1]
        sorted_eig_vals = eig_vals[sorted_i]
        sorted_eig_vecs = eig_vecs[:,sorted_i]
        
        #selecting the first n components
        n_eig_vecs = sorted_eig_vecs[:,:n]
        
        # #transforming data to new labels
        data_set_transformed = np.dot(n_eig_vecs.transpose(), df_centered.transpose()).transpose()
        # print(data_set_transformed)

        headings = []
        for i in range(n):
            #creating a list of headings in data frame
            heading = "PC" + str(i+1)
            headings += [heading] 

            #printing the explained variance by each component
            # perc = sorted_eig_vals[i] / np.sum(sorted_eig_vals)
            # msg = heading + " accounts for " +str(perc)+ "% of the variance in the data"
            # print(msg)

        pc_vals = pd.DataFrame(data_set_transformed, index = target.index ,columns = headings)
        # print(pc_vals)

        pc_df = pd.concat([target, pc_vals] , axis = 1)
        # print(pc_df)

        self.df = pc_df
        # print(self.df)

    def kpca(self, n):
        '''
        Kernal Principle component analysis to reduce dimensions
        ---------- Params ---------- 
        n: int
            Number of principal components to return    
        ---------- Returns ---------- 
        void, function transforms self.df
        '''
        self.normalisation()
        df = self.df.drop(columns=['Close'])
        target = pd.DataFrame(self.df['Close'])

        gamma = 15 #fload for kernal function

        # Calculate pairwise squared Euclidean distances
        sq_dists = sp.spatial.distance.pdist(df, 'sqeuclidean')
        # print(sq_dists)

        # Convert pairwise distances into a square matrix.
        mat_sq_dists = sp.spatial.distance.squareform(sq_dists)    

        # Compute the symmetric kernel matrix.
        K = sp.exp(-gamma * mat_sq_dists)

        # Centering kernel matrix
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # Obtaining eigenpairs from the centered kernel matrix
        eigvals, eigvecs = np.linalg.eigh(K)
        sorted_eig_vals = eigvals[::-1]
        sorted_eig_vecs = eigvecs[:, ::-1]

        # Collect the first i eigenvectors
        data_set_transformed = np.column_stack([sorted_eig_vecs[:, i] for i in range(n)])    

        headings = []
        for i in range(n):
            #creating a list of headings in data frame
            heading = "KPC" + str(i+1)
            headings += [heading] 

            #printing the explained variance by each component
            # perc = sorted_eig_vals[i] / np.sum(sorted_eig_vals)
            # msg = heading + " accounts for " +str(perc)+ "% of the variance in the data"
            # print(msg)

        pc_vals = pd.DataFrame(data_set_transformed, index = target.index ,columns = headings)
        # print(pc_vals)

        pc_df = pd.concat([target, pc_vals] , axis = 1)
        # print(pc_df)

        self.df = pc_df
        # print(self.df)

    def lvf(self, n):
        '''
        Low variance filter to reduce dimensions
        ---------- Params ---------- 
        n: int
            Number of principal components to return    
        ---------- Returns ---------- 
        void, function transforms self.df
        '''
        self.normalisation()
        df = self.df.drop(columns=['Close'])

        vars = df.var().sort_values(ascending=False)
        vars = vars[:n].index.tolist()
        labels = ['Close'] + vars

        self.df = self.df[labels]