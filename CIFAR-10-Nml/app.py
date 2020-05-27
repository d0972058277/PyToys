import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers 
# 從資料夾中的preprocessing.py檔案中Import parse_aug_fn和parse_fn函數
from preprocessing import parse_aug_fn, parse_fn

# 讀取數據並分析
# 載入Cifar10數據集：
# 將train Data重新分成1:9等分，分別分給valid data, train data
valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 90])
# 取得訓練數據，並順便讀取data的資訊
train_data, info = tfds.load("cifar10", split='train[10%:]', with_info=True, data_dir='/home/share/dataset/tensorflow-datasets')
# 取得驗證數據
valid_data = tfds.load("cifar10", split='train[:10%]', data_dir='/home/share/dataset/tensorflow-datasets')
# 取得測試數據
test_data = tfds.load("cifar10", split=tfds.Split.TEST, data_dir='/home/share/dataset/tensorflow-datasets')

# Dataset 設定
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
batch_size = 64  # 批次大小
train_num = int(info.splits['train'].num_examples / 10) * 9  # 訓練資料數量

train_data = train_data.shuffle(train_num)  # 打散資料集
# 載入預處理「 parse_aug_fn」function，cpu數量為自動調整模式
train_data = train_data.map(map_func=parse_aug_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小並將prefetch模式開啟(暫存空間為自動調整模式)
train_data = train_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

# 載入預處理「 parse_fn」function，cpu數量為自動調整模式
valid_data = valid_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小並將prefetch模式開啟(暫存空間為自動調整模式)
valid_data = valid_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

# 載入預處理「 parse_fn」function，cpu數量為自動調整模式
test_data = test_data.map(map_func=parse_fn, num_parallel_calls=AUTOTUNE)
# 設定批次大小並將prefetch模式開啟(暫存空間為自動調整模式)
test_data = test_data.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

def build_and_train_model(run_name, init):
    """
    run_name:傳入目前執行的任務名子
    init:傳入網路層初始化化的方式
    """
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init)(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=init)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=init)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=init)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_initializer=init)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    # 建立網路模型(將輸入到輸出所有經過的網路層連接起來)
    model = keras.Model(inputs, outputs)
    
    # 儲存訓練記錄檔
    logfiles = 'lab5-logs/{}-{}'.format(run_name, init.__class__.__name__)
    model_cbk = keras.callbacks.TensorBoard(log_dir=logfiles, 
                                            histogram_freq=1)
    # 儲存最好的網路模型權重
    modelfiles = model_dir + '/{}-best-model.h5'.format(run_name)
    model_mckp = keras.callbacks.ModelCheckpoint(modelfiles, 
                                                 monitor='val_categorical_accuracy', 
                                                 save_best_only=True, 
                                                 mode='max')
    
    # 設定訓練使用的優化器、損失函數和指標函數
    model.compile(keras.optimizers.Adam(), 
               loss=keras.losses.CategoricalCrossentropy(), 
               metrics=[keras.metrics.CategoricalAccuracy()])
    
    # 訓練網路模型
    model.fit(train_data,
              epochs=100, 
              validation_data=valid_data,
              callbacks=[model_cbk, model_mckp])

session_num = 1
# 設定儲存權重目錄
model_dir = 'lab5-logs/models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 設定要測試的三種初始化方法
weights_initialization_list = [initializers.RandomNormal(0, 0.01),
                        initializers.glorot_normal(),
                        initializers.he_normal()]

for init in weights_initialization_list:
    print('--- Running training session %d' % (session_num))
    run_name = "run-%d" % session_num
    build_and_train_model(run_name, init)  # 創建和訓練網路
    session_num += 1

# 比較三種初始化的訓練結果
model_1 = keras.models.load_model('lab5-logs/models/run-1-best-model.h5')
model_2 = keras.models.load_model('lab5-logs/models/run-2-best-model.h5')
model_3 = keras.models.load_model('lab5-logs/models/run-3-best-model.h5')
loss_1, acc_1 = model_1.evaluate(test_data)
loss_2, acc_2 = model_2.evaluate(test_data)
loss_3, acc_3 = model_3.evaluate(test_data)

loss = [loss_1, loss_2, loss_3]  
acc = [acc_1, acc_2, acc_3]

dict = {"Loss": loss,  
        "Accuracy": acc}

df = pd.DataFrame(dict)
df

# 建立網路模型，這邊使用到以下幾種網路層：
# - keras.Input：輸入層(輸入影像大小為32x32x3)
# - layers.Conv2D：卷積層(使用3x3大小的kernel)
# - layers.BatchNormalization：BatchNormalization層(使用預設參數)
# - layers.ReLU：ReLU激活函數層(使用在BatchNormalization層之後)
# - layers.MaxPool2D：池化層(對特徵圖下採樣)
# - layers.Flatten：扁平層(特徵圖轉成一維Tensor)
# - layers.Dropout：Dropout層(每次訓練隨機丟棄50%網路)
# - layers.Dense：全連接層(隱藏層使用ReLU激活函數，輸出層使用Softmax激活函數)
# 因為大部分激活函數都會在BatchNormalization之後，所以這邊的搭建與前幾個model有些差別。
inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, (3, 3))(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(256, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(128, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Conv2D(64, (3, 3))(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Flatten()(x)
x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)
# 建立網路模型(將輸入到輸出所有經過的網路層連接起來)
model_4 = keras.Model(inputs, outputs, name='model-4')
model_4.summary()  # 顯示網路架構

# 儲存訓練記錄檔
log_dir = os.path.join('lab5-logs', 'run-4-batchnormalization')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# 儲存最好的網路模型權重
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/run-4-best-model.h5', 
                                             monitor='val_categorical_accuracy', 
                                             save_best_only=True, 
                                             mode='max')

model_4.compile(keras.optimizers.Adam(), 
               loss=keras.losses.CategoricalCrossentropy(), 
               metrics=[keras.metrics.CategoricalAccuracy()])

history_4 = model_4.fit(train_data,
                        epochs=100, 
                        validation_data=valid_data,
                        callbacks=[model_cbk, model_mckp])

model_4 = keras.models.load_model('lab5-logs/models/run-4-best-model.h5')
loss, acc = model_4.evaluate(test_data)
print('\nModel-4 Accuracy: {}%'.format(acc))

model_1 = keras.models.load_model('lab5-logs/models/run-1-best-model.h5')
model_2 = keras.models.load_model('lab5-logs/models/run-2-best-model.h5')
model_3 = keras.models.load_model('lab5-logs/models/run-3-best-model.h5')
model_4 = keras.models.load_model('lab5-logs/models/run-4-best-model.h5')
loss_1, acc_1 = model_1.evaluate(test_data)
loss_2, acc_2 = model_2.evaluate(test_data)
loss_3, acc_3 = model_3.evaluate(test_data)
loss_4, acc_4 = model_4.evaluate(test_data)

loss = [loss_1, loss_2, loss_3, loss_4]  
acc = [acc_1, acc_2, acc_3, acc_4]

dict = {"Loss": loss,  
        "Accuracy": acc}

df = pd.DataFrame(dict)
df