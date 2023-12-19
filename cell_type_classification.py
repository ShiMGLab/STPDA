import pandas as pd
import numpy as np
from keras import backend as K
from keras import Input, Model
from keras.layers import Dense, Flatten,Lambda, BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.merge import add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score,recall_score,precision_score # 导入包
learning_rate = 0.0001
from keras.regularizers import l2
from arma_conv import ARMAConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.utils import normalized_laplacian
import os
import matplotlib
matplotlib.use('Agg')
import  tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import pickle
current_path = os.path.abspath('.')
# Parameters
l2_reg = 0.0000001  # Regularization rate for l2
batch_size = 32  # Batch size
epochs = 100 # Number of training epochs
es_patience = 50
channels=1024#you can choose diffenent channels to get better porfermence
import os
# 创建十折交叉验证对象
current_path = os.path.abspath('.')
with open(current_path+'your adjacency matrix', 'rb') as fp:
    adj = pickle.load( fp)
fltr = normalized_laplacian(adj)
gene_expression = pd.read_csv(current_path+'your gene expression')
gene_number=gene_expression.shape[0]
features=gene_expression.div(gene_expression.sum(axis=1)+1, axis='rows')*gene_number
features=np.array(features)
kf = KFold(n_splits=5,shuffle=True,random_state=42)
  # 调用split方法切分数据

# 将数据集拆分成十个子集
labels=pd.read_csv("your cell type lables")
labels=np.array(labels)
train_mask, test_mask = [False] * features.shape[0], [False] * features.shape[0]
total_acc=0
total_precision=0
total_recall=0
total_f1=0
#train_indices,test_indices=train_test_split(np.arange(len(labels)),test_size=0.2,stratify=labels,random_state=1)
for train_index , test_index in kf.split(features):
        for i in train_index:
            train_mask[i] = True
        for i in test_index:
            test_mask[i] = True
        train_mask=np.array(train_mask)
        test_mask=np.array(test_mask)
        N = features.shape[1]
        n_out = 10  # Dimension of the target
        # Model definition
        X_in = Input(shape=(N, ))
        A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))
        graph_conv1 = ARMAConv(channels, activation='relu', kernel_regularizer=l2(l2_reg), use_bias=True)([X_in, A_in])
        graph_conv2 = ARMAConv(channels, activation='relu', kernel_regularizer=l2(l2_reg), use_bias=True)([graph_conv1, A_in])
        graph_conv3 = BatchNormalization()(graph_conv2)
        graph_conv4=tf.reshape(graph_conv3,[features.shape[0],channels,1])
        lstm_1 = LSTM(channels, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(graph_conv4)
        lstm_1b = LSTM(channels, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
            graph_conv4)
        reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1))(lstm_1b)
        lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
        lstm1_merged = BatchNormalization()(lstm1_merged)
        fc = Flatten()(lstm1_merged)
        fc = Dense(512, activation='relu')(graph_conv3)
        output = Dense(n_out, activation='softmax')(fc)
        # Build model
        model = Model(inputs=[X_in, A_in], outputs=output)
        # optimizer = Adam
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy']
        )
        model.fit([features,fltr],labels,sample_weight=train_mask,epochs=100,batch_size=913)
        test_loss,test_acc=model.evaluate([features,fltr],labels,sample_weight=test_mask,batch_size=913,verbose=1)
        print(f'Test Accuracy:{test_acc*100:.2f}%')
        y_predict = model.predict(features, batch_size=913)
        predict_labels=np.argmax(y_predict[test_index],axis=1)# 五种方法
        true_label=labels[test_index].reshape(1,len(test_index))
        true_label=true_label.tolist()
        correct=np.sum(predict_labels==true_label)
        total_sample=len(predict_labels)
        acc=correct/total_sample
        total_acc = total_acc + acc
        precision = precision_score(true_label[0], predict_labels, average='macro')  # 'macro'表示对所有类别的精确率进行简单平均
        total_precision = total_precision + precision
        # 计算召回率
        recall = recall_score(true_label[0],  predict_labels, average='macro')
        total_recall=total_recall+recall# 'macro'表示对所有类别的召回率进行简单平均
        # 计算F1分数
        f1 = f1_score(true_label[0],  predict_labels, average='macro')
        total_f1=total_f1+f1# 'macro'表示对所有类别的F1分数进行简单平均


print(total_acc/5,total_precision/5,total_recall/5,total_f1/5)




