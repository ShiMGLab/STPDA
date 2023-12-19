import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import spektral
from scipy import sparse
import pickle
import scipy.linalg
from sklearn.metrics.pairwise import euclidean_distances
####################  get the whole training dataset

current_path = os.path.abspath('.')
cell_position = pd.read_csv(current_path+"your cell location information")
############# get adjacent matrix,
distance_list_list = []
distance_list_list_2 = []
distance_list = []
for j in range(cell_position.shape[0]):
    for i in range(cell_position.shape[0]):
        if i != j:
            distance_list.append(np.linalg.norm(cell_position.iloc[j][['X', 'Y']] - cell_position.iloc[i][['X', 'Y']]))
distance_list_list = distance_list_list + distance_list
distance_list_list_2.append(distance_list)
distance_array = np.array(distance_list_list)
for threshold in [3]:#[1,2,3,4,5,6,7,8,9,10]:
    print(threshold)
    num_big = np.where(distance_array<threshold)[0].shape[0]
    distance_matrix_threshold_I_list = []
    distance_matrix_threshold_W_list = []
    from sklearn.metrics.pairwise import euclidean_distances
    distance_matrix = euclidean_distances(cell_position[['X', 'Y']], cell_position[['X', 'Y']])  # 欧式距离
    distance_matrix_threshold_I = np.zeros(distance_matrix.shape)  # 0矩阵
    print(distance_matrix_threshold_I.shape)
    distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
    for i in range(distance_matrix_threshold_I.shape[0]):
        for j in range(distance_matrix_threshold_I.shape[1]):
            if distance_matrix[i, j] <= threshold and distance_matrix[i, j] > 0:
                distance_matrix_threshold_I[i, j] = 1
                distance_matrix_threshold_W[i, j] = distance_matrix[i, j]
    distance_matrix_threshold_I_list.append(distance_matrix_threshold_I)
    distance_matrix_threshold_W_list.append(distance_matrix_threshold_W)
    distance_matrix_threshold_I_N = spektral.utils.normalized_adjacency(distance_matrix_threshold_I,
                                                                        symmetric=True)
    # distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I) ## do not normalize adjcent matrix
    distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I_N)
    print(distance_matrix_threshold_I.shape)
    distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)  # 压缩矩阵
    with open(current_path + 'your filepath' , 'wb') as fp:
        pickle.dump(distance_matrix_threshold_I_N_crs, fp)



