import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')

import os
current_path = os.path.abspath('.')
ligand_list = pd.read_csv(current_path+'your ligand data',header  = None)
receptor_list = pd.read_csv(current_path+'your receptor data',header  = None)
ligand_receptor_pairs_list = pd.read_csv(current_path+'your ligand_receptor_pairs data',header  = None,sep ='\t')

#####################
expression_matrix = pd.read_csv(current_path+'your gene expression data')
gene_number=expression_matrix.shape[0]
expression_matrix_normalization =expression_matrix.div(expression_matrix.sum(axis=1)+1, axis='rows')*gene_number
expression_matrix_normalization.columns =[i.lower() for i in list(expression_matrix_normalization)] ## gene expression normalization
#######################

gene_list =[i.lower() for i in list(expression_matrix)]#10000gene


not_ligand_receptor_list = [i for i in gene_list if i not in list(ligand_list.iloc[:,0]) and i not in list(receptor_list.iloc[:,0])]#gene is not in receptor_list and not in ligand_list
ligand_gene_list = [i for i in gene_list if i in list(ligand_list.iloc[:,0])]#gene in gene_list and ligand_list
receptor_gene_list = [i for i in gene_list if i in list(receptor_list.iloc[:,0])]#gene in gene_list and receptor_list
count = 0
ligand_receptor_genes = defaultdict(list)#ligand and recertor all in gene_list
for ligand_receptor_pairs_index in range(ligand_receptor_pairs_list.shape[0]):
    ligand, receptor =  ligand_receptor_pairs_list.iloc[ligand_receptor_pairs_index]
    if ligand in gene_list and receptor in gene_list:
        ligand_receptor_genes[ligand].append(receptor)
        count = count + 1
# generate training dataset containing both postive and negative samples
def generate_ligand_receptor_pairs (ligand_receptor_original,sub_ligand_list, sub_receptor_list,expression_matrix_normalization):
    ligand_receptor = defaultdict(list)
    for ligand in ligand_receptor_original.keys():
        if ligand in sub_ligand_list:
            for receptor in ligand_receptor_original[ligand]:
                if receptor in sub_receptor_list:
                    ligand_receptor[ligand].append(receptor)
    import random
    random.seed(0)
    count = 0
    gene_pairs_list= []
    x_data = []
    y_data = []
    sub_ligand_list_gene = list(ligand_receptor.keys())
    for ligand in sub_ligand_list_gene :
        for receptor in ligand_receptor[ligand]:
            gene_pairs_list.append(ligand + '\t' + receptor)
            cell_ligaand_receptor_expression = np.array(expression_matrix_normalization[[ligand, receptor]]) # postive sample
            x_data.append(cell_ligaand_receptor_expression)
            y_data.append(1)
            ############## get negative samples
            non_pair_receptor_list = [i for i in sub_receptor_list if i not in ligand_receptor[ligand]]
            random.seed(count)
            random_receptor = random.sample(non_pair_receptor_list, 1)[0]
            gene_pairs_list.append(ligand + '\t' + random_receptor)
            ligand_receptor_expression = np.array(expression_matrix_normalization[[ligand, random_receptor]])
            x_data.append(ligand_receptor_expression)
            y_data.append(0)
            count = count + 1
    ligand_record = sub_ligand_list_gene[0]
    gene_pairs_index = [0]
    count = 0
    for gene_pair in gene_pairs_list:
        ligand = gene_pair.split('\t')[0]
        if ligand == ligand_record:
            count = count + 1
        else:
            gene_pairs_index.append(count)
            ligand_record = ligand
            count = count + 1
    gene_pairs_index.append(count)
    x_data_array = np.array(x_data)
    y_data_array = np.array(y_data)
    gene_pairs_list_array = np.array(gene_pairs_list)
    gene_pairs_index_array = np.array(gene_pairs_index)
    return (x_data_array,y_data_array,gene_pairs_list_array,gene_pairs_index_array) ## x data, y data, gene pair name, index to separate pairs by ligand genes


# ten fold cross validation data generation
ligand_list = ligand_gene_list
receptor_list = receptor_gene_list
import random
random.seed(1)
ligand_list_random = random.sample(ligand_list,len(ligand_gene_list))#随机
random.seed(1)
receptor_list_random = random.sample(receptor_list,len(receptor_gene_list))
for test_indel in range(1,11): ################## ten fold cross validation
    print (test_indel)
    ######### completely separate ligand and recpetor genes as mutually  exclusive train and test set
    whole_ligand_index = [i for i in range(len(ligand_list_random))]
    test_ligand = [i for i in range (int(np.ceil((test_indel-1)*0.1*len(ligand_list_random))),int(np.ceil(test_indel*0.1*len(ligand_list_random))))]
    train_ligand= [i for i in whole_ligand_index if i not in test_ligand]
    whole_receptor_index = [i for i in range(len(receptor_list_random))]
    test_receptor = [i for i in range(int(np.ceil((test_indel - 1) * 0.1 * len(receptor_list_random))),int(np.ceil(test_indel * 0.1 * len(receptor_list_random))))]
    train_receptor = [i for i in whole_receptor_index if i not in test_receptor]
    x_train_data_array, y_train_data_array, gene_pairs_list_train_array, gene_pairs_train_index_array = generate_ligand_receptor_pairs (ligand_receptor_genes,np.array(ligand_list_random)[train_ligand], np.array(receptor_list_random)[train_receptor],expression_matrix_normalization)
    x_test_data_array, y_test_data_array, gene_pairs_list_test_array, gene_pairs_index_test_array = generate_ligand_receptor_pairs(ligand_receptor_genes, np.array(ligand_list_random)[test_ligand], np.array(receptor_list_random)[test_receptor],expression_matrix_normalization)
    if not os.path.isdir(current_path + '/ten_fold_crossover/'):
        os.makedirs(current_path + '/ten_fold_crossover/')
    np.save(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_x_data_array.npy', x_train_data_array)
    np.save(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_y_data_array.npy', y_train_data_array)
    np.save(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_gene_pairs_list_array.npy', gene_pairs_list_train_array)
    np.save(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_gene_pairs_index_array.npy', gene_pairs_train_index_array)
    np.save(current_path+'/ten_fold_crossover/' + str(test_indel) + '_test_x_data_array.npy',x_test_data_array)
    np.save(current_path+'/ten_fold_crossover/' + str(test_indel) + '_test_y_data_array.npy',y_test_data_array)
    np.save(current_path+'/ten_fold_crossover/' + str(test_indel) + '_test_gene_pairs_list_array.npy',gene_pairs_list_test_array)
    np.save(current_path+'/ten_fold_crossover/' + str(test_indel) + '_test_gene_pairs_index_array.npy',gene_pairs_index_test_array)