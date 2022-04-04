import numpy as np
class Dimensionality_reduction: 

    def pca(data_set, n):
        
        #centering
        data_set_centered = data_set - np.mean(data_set, axis = 0)
        
        #calculating covariance matrix, eigen matrix and eigen vectors
        cov_matrix = np.cov(data_set_centered, rowvar = False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        
        #sorting eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        
        #selecting the first n components
        n_eigenvectors = sorted_eigenvectors[:,0:n]
        
        #transforming data to new labels
        data_set_transformed = np.dot(n_eigenvectors.transpose(), data_set_centered.transpose()).transpose()
        
        return data_set_transformed

    def kpca(data_set, n):
        data_set_transformed = data_set
        return data_set_transformed