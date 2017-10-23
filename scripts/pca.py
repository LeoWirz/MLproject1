import numpy as np

def PCA(dataset, num_output_features):
    
    #number of features
    num_features = dataset.shape[1]

    # mean of each column
    mean_vector = dataset.mean(axis=0)
    
    # scatter matrix
    scatter_matrix = np.zeros((num_features,num_features))
    for i in range(dataset.shape[1]):
        scatter_matrix += (dataset[i,:].reshape(num_features,1) - mean_vector).dot((dataset[i,:].reshape(num_features,1) - mean_vector).T)
        
    # eigen values and vectors
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
    
    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:,i].reshape(1,num_features).T
        

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # eigen vectors matrix with number of final feature needed  
    matrix_w = np.hstack([[eig_pairs[i][1] for i in range(num_output_features)]])
    
    transformed = dataset.dot(matrix_w.T)
    
    return(transformed)