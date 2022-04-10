import numpy as np
import pandas as pd
# from sklearn.decomposition import PCA

import scipy as sp

class Dimensionality_reduction: 

    def __init__(self, df):
        self.df = df

    # def skpca(self, n):
    #     target = pd.DataFrame(self.df['Close'])
    #     X = self.df
    #     pca = PCA(n_components=n)
    #     pca.fit(X)
    #     X = pca.transform(X)
    #     # print(X)

    #     headings = []
    #     for i in range(n):
    #         heading = "PC" + str(i+1)
    #         headings += [heading]
    #     # print(headings)
    #     pc_vals = pd.DataFrame(X, index = target.index ,columns = headings)
    #     # print(pc_vals)

    #     pc_df = pd.concat([target, pc_vals] , axis = 1)
    #     print(pc_df)


    def pca(self, n):
        '''
        Principle component analysis to reduce dimensions
        ---------- Params ---------- 
        n: int
            Number of principal components to return    
        ---------- Returns ---------- 
        void, function transforms self.df
        '''
        print("Applying PCA to data frame...")
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
            perc = sorted_eig_vals[i] / np.sum(sorted_eig_vals)
            msg = heading + " accounts for " +str(perc)+ "% of the variance in the data"
            print(msg)

        pc_vals = pd.DataFrame(data_set_transformed, index = target.index ,columns = headings)
        # print(pc_vals)

        pc_df = pd.concat([target, pc_vals] , axis = 1)
        # print(pc_df)

        self.df = pc_df
        # print(self.df)

    def kpca(self, n):
        
        df = self.df.drop(columns=['Close'])
        target = pd.DataFrame(self.df['Close'])

        gamma = 15 #fload

        # Calculate pairwise squared Euclidean distances
        # in the MxN dimensional dataset.
        sq_dists = sp.spatial.distance.pdist(df, 'sqeuclidean')
        # print(sq_dists)

        # Convert pairwise distances into a square matrix.
        mat_sq_dists = sp.spatial.distance.squareform(sq_dists)    
        # Compute the symmetric kernel matrix.
        K = sp.exp(-gamma * mat_sq_dists)

        # Center the kernel matrix.
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # Obtaining eigenpairs from the centered kernel matrix
        # scipy.linalg.eigh returns them in ascending order
        eigvals, eigvecs = np.linalg.eigh(K)
        sorted_eig_vals = eigvals[::-1]
        sorted_eig_vecs = eigvecs[:, ::-1]

        # Collect the top k eigenvectors (projected examples)
        data_set_transformed = np.column_stack([sorted_eig_vecs[:, i] for i in range(n)])    

        headings = []
        for i in range(n):
            #creating a list of headings in data frame
            heading = "KPC" + str(i+1)
            headings += [heading] 

            #printing the explained variance by each component
            perc = sorted_eig_vals[i] / np.sum(sorted_eig_vals)
            msg = heading + " accounts for " +str(perc)+ "% of the variance in the data"
            print(msg)

        pc_vals = pd.DataFrame(data_set_transformed, index = target.index ,columns = headings)
        # print(pc_vals)

        pc_df = pd.concat([target, pc_vals] , axis = 1)
        # print(pc_df)

        self.df = pc_df
        # print(self.df)
        
        # return X_pc

        # data_set_transformed = data_set
        
        # return data_set_transformed