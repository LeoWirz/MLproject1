def featuring(features):
    # replace -999 by nan
    temp_feat = np.copy(features)
        
    print("replacing -999 with median")
    for x in temp_feat:
        np.place(x, x == -999, np.nan)
        median = np.nanmedian(x, axis=0)    
        where_are_NaNs = np.isnan(x)
        x[where_are_NaNs] = median
        
    print("applying function")
    stacked_feat = np.hstack([temp_feat] 
                             + [np.power(temp_feat,2)] 
                             + [np.absolute(temp_feat)**0.5]
                             + [np.arctan(temp_feat)]
                             + [add_pairwise_products(temp_feat)]
                             + [np.log(np.absolute(temp_feat) + 1)])
    #inplace_map(np.nan_to_num, stacked_feat)

    print("standardize")
    return standardize(stacked_feat)[0]
    

def inplace_map(f, x):
    for i, v in enumerate(x):
        x[i] = f(v)
        
def add_pairwise_products(x):
    return np.hstack(
        (x,
        np.column_stack([x[:,i] * x[:,j] 
        for i in range(0,x.shape[1]) 
        for j in range(0,x.shape[1]) if i < j])))