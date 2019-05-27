# mnist attention
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.utils import to_categorical
import scipy.io as sio  
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
TIME_STEPS = 200
INPUT_DIM = 1
lstm_units = 16
# import os 
# os.environ("CUDA_VISIBLE_DEVICES")="0"


def getDataindiafrom_mat():
    datafilepath="Indian_pines_corrected.mat"
    labelfilepath="Indian_pines_gt.mat"
    
    a=sio.loadmat(datafilepath)
    aa=sio.loadmat(labelfilepath)
    d=a['indian_pines_corrected']#d.shape you can view the a,d is[145,145,200]
    dOri=a['indian_pines_corrected']
    #im=Image.fromarray(b[:,:,1])
    #im.show()
    l=aa['indian_pines_gt']#label [145*145]
    d=np.float32(d)#数据【145-145*200】
    
    
    
    d /= d.max();#归一化normarlize
    
    
    
    
    #data=np.empty((9715,200),dtype="float32")#21025=145*145
    #label=np.empty((9715),dtype="int32")
    
    dataNormal=np.empty((10249,1,200),dtype="float32")#21025=145*145 ; 其中的0标签是没有标记的标签,有10776个，要剔除
    dataOringin=np.empty((10249,1,200),dtype="float32")
    label=np.empty((10249),dtype="int32")
    
    
    indexofclass=np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))#class number
    dictofclass={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
    #运行完之后的值{0: 46, 1: 1428, 2: 830, 3: 237, 4: 483, 5: 730, 6: 28, 7: 478, 8: 20, 9: 972, 10: 2455, 11: 593, 12: 205, 13: 1265, 14: 386, 15: 93}
    
    
    
    
    index=0
    for i in range(145):#find the no.14 class
        for j in range(145):        
            if (l[i,j]!=0):           
                dataNormal[index,0,:]=d[i,j,:]#获得数据
                dataOringin[index,0,:]=dOri[i,j,:]#获得原始数据
                label[index]=l[i,j]-1#为了后续训练方便，挑出的像素标签相应减1
                #print(i,j,l[i,j],label[index])
                dictofclass[label[index]] += 1 #相应类别数量加1            
                index += 1
    
    
    
    return dataNormal,dataOringin,label


dataNormal,dataOringin,label=getDataindiafrom_mat()
dataOringin=dataOringin.reshape(10249,200,1)
label=label.reshape(10249,)

#data = np.load('./FGK_fluxes_35544.npy')
# data=data.reshape((35544,1,2000))
#label = np.load('./FGK_subclass_35544.npy')-3
num_train = 10512
longth = 200
n_class = 16




X_train,X_test,Y_train,Y_test = train_test_split(dataOringin,label,test_size = 0.30,random_state = 0)
# X_test_= X_test.reshape([label.shape[0] - num_train, longth,1])

# (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
# X_train = X_train.reshape(-1, 28, 28) / 255.
# X_test = X_test.reshape(-1, 28, 28) / 255.
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# first way attention
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

# build RNN model with attention
inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
drop1 = Dropout(0.5)(inputs)
# conv = Conv2D(16, (1, 3), padding='same')
# conved_a = conv(a)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.5)(attention_flatten)
output = Dense(16, activation='sigmoid')(drop2)
model = Model(inputs=inputs, outputs=output)

# second way attention
# inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
# units = 32
# activations = LSTM(units, return_sequences=True, name='lstm_layer')(inputs)
#
# attention = Dense(1, activation='tanh')(activations)
# attention = Flatten()(attention)
# attention = Activation('softmax')(attention)
# attention = RepeatVector(units)(attention)
# attention = Permute([2, 1], name='attention_vec')(attention)
# attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
# out_attention_mul = Flatten()(attention_mul)
# output = Dense(10, activation='sigmoid')(out_attention_mul)
# model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

print('Training------------')
Y_train = to_categorical(Y_train)
model.fit(X_train, Y_train, epochs=100, batch_size=200)

print('Testing--------------')
Y_test = to_categorical(Y_test)
loss, accuracy = model.evaluate(X_test, Y_test)
model.save('my_model.h5')

print('test loss:', loss)
print('test accuracy:', accuracy)