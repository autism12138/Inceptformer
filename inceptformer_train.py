from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from cyr_net import cnnformer
# from Unet import cnnformer
import tensorflow as tf
import cyr_data_generator
import scipy.io as scio
from scipy import signal
from keras.models import Model
from keras.layers import Input
import numpy as np
import os
import matplotlib.pylab as plt


# get the filtered EEG-data, label and the start time of each trial of the dataset
def get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, path):
    # read the data
    data = scio.loadmat(path)
    data_1 = data['data']
    # get the EEG data of selected 9 electrodes
    c1 = [47, 53, 54, 55, 56, 57, 60, 61, 62]
    train_data = data_1[c1, :, :, :]
    # 2024/08/03/18/16
    block_data_list1 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn11, wn21], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k, :, j, i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list1.append(np.array(target_data_list))
    # get the filtered EEG-data with six-order Butterworth filter of the second sub-filter
    block_data_list2 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn12, wn22], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k, :, j, i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list2.append(np.array(target_data_list))
    # get the filtered EEG-data with six-order Butterworth filter of the third sub-filter
    block_data_list3 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn13, wn23], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k, :, j, i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list3.append(np.array(target_data_list))
    return np.array(block_data_list1), np.array(block_data_list2), np.array(block_data_list3)


# 定义学习率调度函数
def lr_schedule(epoch, lr):
    if epoch > 10:
        lr = lr * tf.math.exp(-0.1)
    return lr


if __name__ == '__main__':
    # open the GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # %% Setting hyper-parameters
    # ampling frequency after downsampling
    fs = 250
    # the number of the electrode channels
    channel = 9
    # the hyper-parameters of the training process
    train_epoch = 500
    batchsize = 256

    # the filter ranges of the four sub-filters in the filter bank
    f_down1 = 6
    f_up1 = 50
    wn11 = 2 * f_down1 / fs
    wn21 = 2 * f_up1 / fs

    f_down2 = 14
    f_up2 = 50
    wn12 = 2 * f_down2 / fs
    wn22 = 2 * f_up2 / fs

    f_down3 = 22
    f_up3 = 50
    wn13 = 2 * f_down3 / fs
    wn23 = 2 * f_up3 / fs

    mask_rate_list = [0.2]  # [0.0,0.1,0.2,0.3,0.4,0.5]
    t_train_list = [1.0]  # [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    # train_list_sub = [31,32,34,35]

    for sub_selelct in range(1, 36):
        # for sub_selelct in train_list_sub:

        # accuracy_dict = {sub: {block: [] for block in range(6)} for sub in range(1, 36)}
        path = '../benchmark/S%d.mat' % sub_selelct
        data1, data2, data3 = get_train_data(wn11, wn21, wn12, wn22, wn13, wn23, path)

        # accuracy_dict[sub_selelct] = []

        for t_train in t_train_list:
            win_train = int(fs * t_train)
            for mask_rate in mask_rate_list:
                for block_n in range(6):
                    train_list = list(range(6))
                    val_list = [block_n]
                    train_list = [i for i in train_list if (i not in val_list)]
                    # data generator (generate the taining and validation samples of batchsize trials)
                    train_gen = cyr_data_generator.train_datagenerator(batchsize, data1, data2, data3, win_train,
                                                                   train_list, channel, mask_rate)  # , t_train)
                    # 验证数据20240714-14:22
                    val_gen = cyr_data_generator.train_datagenerator(batchsize, [data1[block_n]],
                                                                 [data2[block_n]],
                                                                 [data3[block_n]], win_train, val_list, channel,
                                                                 mask_rate)

                    # %% setting the input of the network
                    input_shape = (channel, win_train, 3)
                    input_tensor = Input(shape=input_shape)
                    # using the CNN-Former model
                    preds = cnnformer(input_tensor)
                    # print("preds:",preds)
                    model = Model(input_tensor, preds)
                    # the path of the saved model and you need to change it
                    model_path = '12.23/former_%3.1fs_%d_mask%3.1f_block%d.h5' % (t_train, sub_selelct, mask_rate, block_n)
                    # 定义学习率调度器
                    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

                    # 定义Early Stopping
                    early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=1)

                    # 定义ReduceLROnPlateau
                    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1)

                    # some hyper-parameters in the training process
                    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True,
                                                       mode='auto')
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    # training, using model.fit or model.fit_generator

                    history = model.fit(
                        train_gen,
                        steps_per_epoch=10,
                        epochs=train_epoch,
                        validation_data=None,
                        validation_steps=1,
                        callbacks=[model_checkpoint, early_stopping, reduce_lr]
                    )
                    # accuracy_dict[sub_selelct][block_n]=history.history['accuracy']