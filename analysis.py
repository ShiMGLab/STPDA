
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
data_augmentation = False
# num_predictions = 20
batch_size = 256
num_classes = 3
epochs = 200
data_augmentation = False
# num_predictions = 20
model_name = 'keras_cnn_trained_model_shallow.h5'
# The data, shuffled and split between train and test sets:
current_path = os.path.abspath('.')


save_dir = os.path.join(os.getcwd(),'your save_name')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    # plt.grid()

mean_fpr = np.linspace(0, 1, 100)
y_test = np.empty([0,1])
y_predict= np.empty([0,1])#创建数组
count_set = [0]
for test_indel in range(1,11):
    X_data_test = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_test_x_data_array.npy')
    Y_data_test = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_test_y_data_array.npy')
    count_setz = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_test_gene_pairs_index_array.npy')
    y_predict_end = np.load(current_path+'/'+str(test_indel)+'your save_name' + '/end_y_predict.npy')
    y_testyz = np.load(current_path+'/'+str(test_indel)+'your save_name' + '/end_y_test.npy')
    y_test = np.concatenate((y_test,y_testyz),axis = 0)
    y_predict = np.concatenate((y_predict, y_predict_end), axis=0)
    count_set = count_set + [i + count_set[-1] if len(count_set)>0 else i for i in count_setz[1:]]
AUC_set =[]
s = open(save_dir+'/AUCs.txt','w')
tprs = []
mean_fpr = np.linspace(0, 1, 100)
#######################################
##################################
fig = plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1])
total_pair = 0
total_auc = 0
    ############
for jj in range(len(count_set)-1):#len(count_set)-1):
    if count_set[jj] < count_set[jj+1]:
        print (test_indel,jj,count_set[jj],count_set[jj+1])
        current_pair = count_set[jj+1] - count_set[jj]
        total_pair = total_pair + current_pair
        y_test_a = y_test[count_set[jj]:count_set[jj+1]]
        y_predict_a = y_predict[count_set[jj]:count_set[jj+1],0]
        # Score trained model.
        fpr, tpr, thresholds = metrics.roc_curve(y_test_a, y_predict_a, pos_label=1)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
            # Print ROC curve
        plt.plot(fpr, tpr, color='0.5', lw=0.001,alpha=.2)
        auc = np.trapz(tpr, fpr)
        s.write(str(jj)+'\t'+str(count_set[jj])+'\t'+str(count_set[jj+1])+'\t'+str(auc) + '\n')
        print('AUC:', auc)
        AUC_set.append(auc)
        total_auc = total_auc + auc * current_pair


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
per_tpr = np.percentile(tprs,[40,50,60],axis=0)
mean_auc = np.trapz(mean_tpr,mean_fpr)
plt.plot(mean_fpr, mean_tpr,'k',lw=3,label = 'mean ROC')
plt.title("{:.4f}".format(mean_auc),fontsize=15)
plt.fill_between(mean_fpr, per_tpr[0,:], per_tpr[2,:], color='g', alpha=.2,label='quantile')
plt.plot(mean_fpr, per_tpr[0,:],'g',lw=3,alpha=.2)
plt.legend(loc='lower right',fontsize=15)
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.grid()
plt.xlabel('FP', fontsize=15)
plt.ylabel('TP', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(save_dir+'/ROCs.pdf')
del fig
fig = plt.figure(figsize=(3, 3))
plt.hist(AUC_set,bins = 50)
plt.savefig(save_dir + '/ROCs.pdf')
del fig
s.close()
fig = plt.figure(figsize=(3, 3))
plt.boxplot(AUC_set)
plt.savefig(save_dir + '/ROCs.pdf')
del fig
############################

###################################################################### PR
##################################
AUC_set =[]
s = open(save_dir+'/AUC_PR.txt','w')
tprs = []
mean_fpr = np.linspace(0, 1, 100)
fig = plt.figure(figsize=(5, 5))
total_pair = 0
total_auc = 0
print (y_predicty.shape)
    ############
for jj in range(len(count_set)-1):#len(count_set)-1):
    if count_set[jj] < count_set[jj+1]:
        print (test_indel,jj,count_set[jj],count_set[jj+1])
        current_pair = count_set[jj+1] - count_set[jj]
        total_pair = total_pair + current_pair
        y_test = y_testy[count_set[jj]:count_set[jj+1]]
        y_predict = y_predicty[count_set[jj]:count_set[jj+1]]
        # Score trained model.
        tpr, fpr, thresholds = metrics.precision_recall_curve(y_test, y_predict)  # , pos_label=1)
        tpr = np.flip(tpr)
        fpr = np.flip(fpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        plt.plot(fpr, tpr, color='0.5', lw=0.001,alpha=.2)
        auc = np.trapz(tpr, fpr)
        s.write(str(jj)+'\t'+str(count_set[jj])+'\t'+str(count_set[jj+1])+'\t'+str(auc) + '\n')
        print('AUC:', auc)
        AUC_set.append(auc)
        total_auc = total_auc + auc * current_pair

mean_tpr = np.mean(tprs, axis=0)

per_tpr = np.percentile(tprs,[40,50,60],axis=0)
mean_auc = np.trapz(mean_tpr,mean_fpr)
plt.plot(mean_fpr, mean_tpr,'k',lw=3,label = 'mean ROC')
plt.title("{:.4f}".format(mean_auc),fontsize=15)
plt.fill_between(mean_fpr, per_tpr[0,:], per_tpr[2,:], color='g', alpha=.2,label='quantile')
plt.plot(mean_fpr, per_tpr[0,:],'g',lw=3,alpha=.2)
plt.legend(loc='lower right',fontsize=15)
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.grid()
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(save_dir +'/ROCs_PR.pdf')
del fig
fig = plt.figure(figsize=(3, 3))
plt.hist(AUC_set,bins = 50)
plt.savefig(save_dir  +'/ROCs_PR.pdf')
del fig
s.close()
fig = plt.figure(figsize=(3, 3))
plt.boxplot(AUC_set)
plt.savefig(save_dir  +'/ROCs_PR.pdf')
del fig



