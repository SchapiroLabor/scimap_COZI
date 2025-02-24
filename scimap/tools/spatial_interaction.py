#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Oct 19 20:03:01 2020
# @author: Ajit Johnson Nirmal

"""
!!! abstract "Short Description"
    `sm.tl.spatial_interaction`: This function quantifies the spatial interactions 
    between cell types, assessing their co-localization beyond random chance, with 
    support for both 2D and 3D datasets. By comparing observed adjacency frequencies 
    to a random distribution, it helps uncover significant cellular partnerships 
    within tissue contexts.
    
## Function
"""

# Import library
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from joblib import Parallel, delayed
import scipy
from functools import reduce
from scipy.spatial import Delaunay


# Function
def spatial_interaction (adata,
                         x_coordinate='X_centroid',
                         y_coordinate='Y_centroid',
                         z_coordinate=None,
                         phenotype='phenotype',
                         method='radius',
                         radius=30,
                         knn=10,
                         permutation=1000,
                         imageid='imageid',
                         subset=None,
                         pval_method='zscore',
                         normalization='total',
                         verbose=True,
                         scaling=False,
                         label='spatial_interaction'):
    """
Parameters:
        adata (anndata.AnnData):  
            Annotated data matrix or path to an AnnData object, containing spatial gene expression data.

        x_coordinate (str, required):  
            Column name in `adata` for the x-coordinates.

        y_coordinate (str, required):  
            Column name in `adata` for the y-coordinates.

        z_coordinate (str, optional):  
            Column name in `adata` for the z-coordinates, for 3D spatial data analysis.

        phenotype (str, required):  
            Column name in `adata` indicating cell phenotype or any categorical cell classification.

        method (str, optional):  
            Method to define neighborhoods: 'radius' for fixed distance, 'knn' for K nearest neighbors and 'delaunay' for Delaunay triangulation.

        radius (int, optional):  
            Radius for neighborhood definition (applies when method='radius').

        knn (int, optional):  
            Number of nearest neighbors to consider (applies when method='knn').

        permutation (int, optional):  
            Number of permutations for p-value calculation.

        imageid (str, required):  
            Column name in `adata` for image identifiers, useful for analysis within specific images.

        subset (str, optional):  
            Specific image identifier for targeted analysis.

        pval_method (str, optional):  
            Method for p-value calculation: 'abs' for absolute difference, 'zscore' for z-score based significance.

        normalization (str, optional):
            Method for normalization: 'total' for total cell count normalization, 'conditional' for conditional normalization (adapted from histocat).
        
        verbose (bool):  
            If set to `True`, the function will print detailed messages about its progress and the steps being executed.

        label (str, optional):  
            Custom label for storing results in `adata.obs`.

Returns:
        adata (anndata.AnnData):  
            Updated `adata` object with spatial interaction results in `adata.obs[label]`.

Example:
        ```python
        
        # Radius method for 2D data with absolute p-value calculation
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    method='radius', radius=50, permutation=1000, pval_method='abs',
                                    label='interaction_radius_abs')
    
        # KNN method for 2D data with z-score based p-value calculation
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    method='knn', knn=15, permutation=1000, pval_method='zscore',
                                    label='interaction_knn_zscore')
    
        # Radius method for 3D data analysis
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    z_coordinate='Z_centroid', method='radius', radius=60, permutation=1000,
                                    pval_method='zscore', label='interaction_3D_zscore')
        
        ```
    """
    
    
    def spatial_interaction_internal (adata_subset,
                                      x_coordinate,
                                      y_coordinate,
                                      z_coordinate,
                                      phenotype,
                                      method,
                                      radius,
                                      knn,
                                      permutation, 
                                      imageid,
                                      subset,
                                      pval_method,
                                      normalization):
        if verbose:
            print("Processing Image: " + str(adata_subset.obs[imageid].unique()))
        
        # Create a dataFrame with the necessary information
        # This is useful for 3D data or 2D data with Z coordinate (multi stacks)
        if z_coordinate is not None:
            if verbose:
                print("Including Z -axis")
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'z': adata_subset.obs[z_coordinate], 'phenotype': adata_subset.obs[phenotype]})
        else:
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'phenotype': adata_subset.obs[phenotype]})

        
        # Select the neighborhood method, knn, radius or delaunay
        # a) KNN method
        if method == 'knn':
            if verbose:
                print("Identifying the " + str(knn) + " nearest neighbours for every cell")
            if z_coordinate is not None:
                tree = BallTree(data[['x','y','z']], leaf_size= 2)
                ind = tree.query(data[['x','y','z']], k=knn, return_distance= False)
            else:
                tree = BallTree(data[['x','y']], leaf_size= 2)
                ind = tree.query(data[['x','y']], k=knn, return_distance= False)
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighbour DF
            neighbours.drop(0, axis=1, inplace=True) # Remove self neighbour
            
        # b) Local radius method
        if method == 'radius':
            if verbose:
                print("Identifying neighbours within " + str(radius) + " pixels of every cell")
            if z_coordinate is not None:
                kdt = BallTree(data[['x','y','z']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y','z']], r=radius, return_distance=False)
            else:
                kdt = BallTree(data[['x','y']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y']], r=radius, return_distance=False)
                
            for i in range(0, len(ind)): ind[i] = np.delete(ind[i], np.argwhere(ind[i] == i))#remove self
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighborhood DF

        # c) Delaunay triangulation method
        if method == 'delaunay':
            if verbose:
                print("Performing Delaunay triangulation to identify neighbours for every cell")
            if z_coordinate is not None:
                points = data[['x', 'y', 'z']].values
            else:
                points = data[['x', 'y']].values

            # Perform Delaunay triangulation
            delaunay = Delaunay(points)

            # Initialize a dictionary to store neighbours
            neighbours_dict = {i: set() for i in range(len(points))}

            # Iterate over each simplex (triangle/tetrahedron) to populate the neighbours dictionary
            for simplex in delaunay.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        neighbours_dict[simplex[i]].add(simplex[j])
                        neighbours_dict[simplex[j]].add(simplex[i])

            # Convert the neighbours dictionary to a list of lists
            neighbours_list = [list(neighbours) for neighbours in neighbours_dict.values()]

            # Ensure each list has the same number of elements by padding with -1 (assuming indices are non-negative)
            max_neigh_len = max(len(neigh) for neigh in neighbours_list)
            neighbours_list_padded = [neigh + [-1] * (max_neigh_len - len(neigh)) for neigh in neighbours_list]

            # Convert to numpy array for consistency with KNN method
            ind = np.array(neighbours_list_padded)

            # Convert to DataFrame for the same output format as the original function
            neighbours = pd.DataFrame(ind.tolist(), index=data.index)

            # Replace -1 with None
            neighbours.replace(-1, None, inplace=True)

        ### END OF NEIGHBORHOOD SELECTION ###
        # Map Phenotypes to Neighbours
        # Loop through (all functionized methods were very slow)
        phenomap = dict(zip(list(range(len(ind))), data['phenotype'])) # Used for mapping
        if verbose:
            print("Mapping phenotype to neighbors")
        for i in neighbours.columns:
            neighbours[i] = neighbours[i].dropna().map(phenomap, na_action='ignore')
            
        # Drop NA
        neighbours = neighbours.dropna(how='all')
        
        # Collapse all the neighbours into a single column
        n = pd.DataFrame(neighbours.stack(), columns = ["neighbour_phenotype"])
        n.index = n.index.get_level_values(0) # Drop the multi index
        
        # Merge with real phenotype
        n = n.merge(data['phenotype'], how='inner', left_index=True, right_index=True)
        
        # Permutation
        if verbose:
            print('Performing '+ str(permutation) + ' permutations')

        #### Permutation ####
        # Set a global seed for reproducibility
        np.random.seed(42)

        # Generate fixed seeds for all permutations
        seeds = np.random.randint(0, 1e6, size=permutation) 

        def permutation_pval (data, seed):
           # Permute the neighbour_phenotype column without affecting the original data structure
            # set seed
            np.random.seed(seed)
            data = data.assign(neighbour_phenotype=np.random.permutation(data['neighbour_phenotype']))
            #print(data)
            k = data.groupby(['phenotype','neighbour_phenotype'],observed=False).size().unstack()#.fillna(0)
            # add neighbour phenotype that are not present to make k a square matrix
            columns_to_add = dict.fromkeys(np.setdiff1d(k.index,k.columns), 0)
            k = k.assign(**columns_to_add)
            total_cell_count = data.reset_index().drop_duplicates(subset=['index', 'phenotype']).groupby('phenotype').size().reindex(k.index, fill_value=0)  # Ensure all categories are included
            data_freq = k.div(total_cell_count, axis = 0)
            data_freq = data_freq.fillna(0).stack().values  # Flatten the matrix
            return data_freq

        def permutation_pval_norm (data, seed):
            # Permute the neighbour_phenotype column without affecting the original data structure
            # set seed
            np.random.seed(seed)
        
            data = data.assign(neighbour_phenotype=np.random.permutation(data['neighbour_phenotype']))
            data_freq = data.groupby(['phenotype','neighbour_phenotype'],observed=False).size().unstack()

            # Remove duplicate interactions (conditional factor)
            data = data.reset_index()
            data = data.drop_duplicates()
            data = data.set_index('index')

            # We noralize the data based on the number of cells of each type 
            normalization_factor = data.groupby(['phenotype', 'neighbour_phenotype'],observed=False).size().unstack()
            data_freq = data_freq/normalization_factor
            data_freq = data_freq.fillna(0).stack().values
            return data_freq

        # Apply permutation functions depending on normalization
        
        # Apply permutation functions depending on normalization
        if normalization == "total":
            final_scores = Parallel(n_jobs=-1)(
                delayed(permutation_pval)(data=n, seed=seeds[i]) for i in range(permutation))
        if normalization == "conditional":
            final_scores = Parallel(n_jobs=-1)(
                delayed(permutation_pval_norm)(data=n, seed=seeds[i]) for i in range(permutation))

        # Permutation results
        perm = pd.DataFrame(final_scores).T
        #print("perm:", perm)
        
        # Consolidate the permutation results
        if verbose:
            print('Consolidating the permutation results')

        # Calculate P value
        # N_freq is the observed frequency of each cell type with each of its neighbours (observed number of interactions)
        if normalization == "total":
            # Calculate interaction frequencies without dropping any categories
            # Normalize based on total cell count
            k = n.groupby(['phenotype','neighbour_phenotype'],observed=False).size().unstack()#.fillna(0)
            # add neighbour phenotype that are not present to make k a square matrix
            columns_to_add = dict.fromkeys(np.setdiff1d(k.index,k.columns), 0)
            k = k.assign(**columns_to_add)
            #total_cell_count = data['phenotype'].value_counts().reindex(k.columns, fillvalue=0).values
            #total_cell_count = data.reset_index().drop_duplicates(subset=['index', 'phenotype']).groupby('phenotype').size().reindex(k.index, fill_value=0)  # Ensure all categories are included
            total_cell_count = data['phenotype'].value_counts()
            n_freq = k.div(total_cell_count, axis = 0)
            n_freq = n_freq.fillna(0).stack()  # Flatten the matrix

        # Normalize n_freq if normalization is conditional
        if normalization == "conditional":
            # Calculate observed interaction frequencies
            data = n.assign(neighbour_phenotype=n['neighbour_phenotype'])
            data_freq = n.groupby(['phenotype', 'neighbour_phenotype'], observed=False).size().unstack()

            # Remove duplicate interactions (conditional factor)
            data = data.reset_index()
            data = data.drop_duplicates()
            data = data.set_index('index')

            normalization_factor = data.groupby(['phenotype', 'neighbour_phenotype'],observed=False).size().unstack()
            data_freq = data_freq / normalization_factor
            n_freq = data_freq.fillna(0).stack()
   
        # permutation with scaling
        if scaling == True:
            perm_scaled = perm.apply(lambda row: 2 * (row - row.min()) / (row.max() - row.min()) - 1, axis=1)
            mean = perm_scaled.mean(axis=1)
            std = perm_scaled.std(axis=1)
            # Initialize a new Series to store scaled `n_freq`
            n_freq_scaled = n_freq.copy()

            # Normalize `n_freq` using the min and max of the corresponding rows in `perm`
            for i in range(len(n_freq_scaled)):
                row_min = perm.iloc[i, :].min()
                row_max = perm.iloc[i, :].max()
                n_freq_scaled.iloc[i] = 2 * (n_freq.iloc[i] - row_min) / (row_max - row_min) - 1
                n_freq = n_freq_scaled
        else:
            mean = perm.mean(axis=1)
            std = perm.std(axis=1)

        # P-value calculation
        if pval_method == 'abs':
            # Calculate the number of times permuted values exceed the observed
            p_values = np.sum(perm >= n_freq.values[:, None], axis=1) / (permutation + 1)
            p_values = p_values[~np.isnan(p_values)].values

        if pval_method == 'zscore':
            z_scores = (n_freq.values - mean) / std        
            z_scores[np.isnan(z_scores)] = 0
            p_values = scipy.stats.norm.sf(abs(z_scores))*2
            p_values = p_values[~np.isnan(p_values)]

        # Compute Direction of interaction (interaction or avoidance)
        direction = ((n_freq.values - mean) / abs(n_freq.values - mean)).fillna(1)


        # DataFrame with the neighbour frequency and P values
        if pval_method == 'abs':
            count = (n_freq.values * direction).values # adding directionallity to interaction
            neighbours = pd.DataFrame({'count': count, 'p_val': p_values}, index=n_freq.index)
            neighbours.columns = [adata_subset.obs[imageid].unique()[0],
                                  'pvalue_' + str(adata_subset.obs[imageid].unique()[0])]
            neighbours = neighbours.reset_index()

        elif pval_method == 'zscore':
            #count = (n_freq.values * direction).values # adding directionality to interaction
            count = n_freq.values
            neighbours = pd.DataFrame({'z_score':z_scores.values,'p_val': p_values, 'count':n_freq}, index = n_freq.index)
            neighbours.columns = ['zscore_' + str(adata_subset.obs[imageid].unique()[0]),
                                  'pvalue_' + str(adata_subset.obs[imageid].unique()[0]),
                                  'count_' + str(adata_subset.obs[imageid].unique()[0])]
            neighbours = neighbours.reset_index()
        
        # Return the results
        return neighbours
          
      
    # subset a particular subset of cells if the user wants else break the adata into list of anndata objects
    if subset is not None:
        adata_list = [adata[adata.obs[imageid] == subset]]
    else:
        adata_list = [adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()]
    
    
    # Apply function to all images and create a master dataframe
    # Create lamda function 
    r_spatial_interaction_internal = lambda x: spatial_interaction_internal (adata_subset=x,
                                                                             x_coordinate=x_coordinate,
                                                                             y_coordinate=y_coordinate,
                                                                             z_coordinate=z_coordinate,
                                                                             phenotype=phenotype,
                                                                             method=method,
                                                                             radius=radius,
                                                                             knn=knn,
                                                                             permutation=permutation,
                                                                             imageid=imageid,
                                                                             subset=subset,
                                                                             pval_method=pval_method,
                                                                             normalization=normalization)

    # Apply function to all images
    all_data = list(map(r_spatial_interaction_internal, adata_list)) # Apply function

    # Merge all the results into a single dataframe    
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['phenotype', 'neighbour_phenotype'], how='outer'), all_data)

    # Add to anndata
    adata.uns[label] = df_merged
    
    # return
    return adata