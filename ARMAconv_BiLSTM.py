
import numpy as np
from keras import backend as K
from keras import Input, Model
from keras.layers import Dense, Flatten,Lambda, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.merge import add
from keras.optimizers import adam_v2
learning_rate = 0.000001
Adam=adam_v2.Adam(learning_rate=learning_rate,beta_2=0.999,beta_1=0.9,epsilon=1e-08)
from keras.regularizers import l2
from arma_conv import ARMAConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils import normalized_laplacian
from keras.utils.vis_utils import plot_model
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
current_path = os.path.abspath('.')
# Parameters
l2_reg = 0.0000001  # Regularization rate for l2
  # Learning rate for SGD
batch_size = 32  # Batch size
epochs = 100 # Number of training epochs
import pickle
with open(current_path+'your adjacency matrix', 'rb') as fp:
    adj = pickle.load( fp)
# adj = np.load('/home/yey3/spatial_nn/processed_data/sourcedata/sy/FOV_0_distance_I_N_crs.npy')
for test_indel in range(1,11): ################## ten fold cross validation
    X_data_train = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_x_data_array.npy')
    Y_data_train = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_y_data_array.npy')
    gene_pair_index_train = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_gene_pairs_list_array.npy')
    count_setx_train = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_train_gene_pairs_index_array.npy')
    X_data_test = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_test_x_data_array.npy')
    Y_data_test = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_test_y_data_array.npy')
    gene_pair_index_test = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_test_gene_pairs_list_array.npy')
    count_set = np.load(current_path+'/ten_fold_crossover/'+str(test_indel)+'_test_gene_pairs_index_array.npy')
    trainX_index = [i for i in range(Y_data_train.shape[0])]
    validation_index = trainX_index[:int(np.ceil(0.2*len(trainX_index)))]
    train_index = trainX_index[int(np.ceil(0.2*len(trainX_index))):]
    X_train, y_train = X_data_train[train_index],Y_data_train[train_index][:,np.newaxis]
    X_val, y_val= X_data_train[validation_index],Y_data_train[validation_index][:,np.newaxis]#增加维度
    X_test, y_test= X_data_test,Y_data_test[:,np.newaxis]
    N = X_train.shape[-2]  # Number of nodes in the graphs
    F = X_train.shape[-1]  # Node features dimensionality
    n_out = y_train.shape[-1]  # Dimension of the target
    fltr = normalized_laplacian(adj)
    # Model definition
    X_in = Input(shape=(N, F))
    A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))
    graph_conv1 = ARMAConv(32, activation='relu', kernel_regularizer=l2(l2_reg), use_bias=True)([X_in, A_in])
    graph_conv1 =ARMAConv(32, activation='relu', kernel_regularizer=l2(l2_reg), use_bias=True)(([graph_conv1, A_in]))
    graph_conv2=BatchNormalization()(graph_conv1)
    lstm_1 = LSTM(32, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(X_in)
    lstm_1b = LSTM(32,return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
        X_in)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)
    lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
    lstm1_merged = BatchNormalization()(lstm1_merged)
    fc = Flatten()(lstm1_merged)
    fc = Dense(512, activation='relu')(fc)
    output = Dense(n_out, activation='sigmoid')(fc)
    # Build model
    model = Model(inputs=[X_in, A_in], outputs=output)
    #optimizer = Adam
    model.compile(optimizer=Adam,loss='binary_crossentropy',metrics=['acc'])
    model.summary()

    plot_model(model, to_file='gcn_LR_spatial_1.png', show_shapes=True)
    save_dir = current_path+'/'+str(test_indel)+'your save_name'+str(epochs)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    history = model.fit(X_train,y_train,batch_size=32,epochs=epochs)
    model_name = 'arma_bilstm.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = model.evaluate(X_test, y_test, verbose=1,batch_size=batch_size)
    y_predict = model.predict(X_test)
    np.save(save_dir + '/end_y_test.npy', y_test)
    np.save(save_dir + '/end_y_predict.npy', y_predict)
    ############################################################################## plot training process

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.savefig(save_dir + '/end_result.pdf')
    ############################################################### 
    #######################################

    #############################################################
    #########################
    y_testy = y_test
    y_predicty = y_predict
    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1])
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    # plt.grid()
    AUC_set = []
    s = open(save_dir + '/divided_interaction.txt', 'w')
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)  # 3068
    for jj in range(len(count_set) - 1):  # len(count_set)-1):
        if count_set[jj] < count_set[jj + 1]:
            print(test_indel, jj, count_set[jj], count_set[jj + 1])
            y_test = y_testy[count_set[jj]:count_set[jj + 1]]
            y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
            # Score trained model.
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # Print ROC curve
            plt.plot(fpr, tpr, color='0.5', lw=0.001, alpha=.2)
            auc = np.trapz(tpr, fpr)
            s.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
            print('AUC:', auc)
            AUC_set.append(auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
    mean_auc = np.trapz(mean_tpr, mean_fpr)
    plt.plot(mean_fpr, mean_tpr, 'k', lw=3, label='median ROC')
    plt.title(str(mean_auc))
    plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='Quartile')
    plt.plot(mean_fpr, per_tpr[0, :], 'g', lw=3, alpha=.2)
    plt.legend(loc='lower right')
    plt.savefig(save_dir + '/ROC.pdf')
    del fig
    fig = plt.figure(figsize=(5, 5))
    plt.hist(AUC_set, bins=50)
    plt.savefig(save_dir + '/ROC_hist.pdf')
    del fig
    s.close()

