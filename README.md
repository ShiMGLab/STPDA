# 1, data prepare
First you need have a konwn receptor list,ligand list,and interacting receptor ligand table 
Second you need Spatial transcriptome data with gene expression data and cell spatial location information 
># 2, Code environment
First you need to install 'python'
Second the python need have 'numpy','pandas','keras','tensorflow','spektral',and so on.
># 3,Usage
3.1 creat adjacency matrix of cells
    `python creat_adjacency_matrix.py`# use this code requires cell spatial location information 
3.2 creat a ten fold cross dataset
     `python data_processing.py`  #use this code you need to prepare a konwn receptor list,ligand list,and interacting receptor ligand table.It also have expression data  and  save it in `ten_fold_crossover` folder.
Users should first set the path as the downloaded folder. 

3.3 Training and test model

   ` python ARMAconv_BiLSTM.py` # use this code to generate normalized laplacian matrix, and then use laplacian matrix and gene expression matrix train to test  model in ten fold cross validation.
3.4 analysis and get performence
    ` analysis.py`
3.5 get cell type
   `python cell_type_classification.py`

 
