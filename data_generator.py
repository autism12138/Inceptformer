from random import sample
import random
import numpy as np
import keras
from keras.utils import to_categorical


# get the training sampels
def train_datagenerator(batchsize, train_data1, train_data2, train_data3, win_train, train_list, channel, mask_rate):
    while True:
        x_train1, x_train2, x_train3, y_train = list(range(batchsize)), list(range(batchsize)), list(
            range(batchsize)), list(range(batchsize))
        # get the list of target stimulus
        target_list = list(range(40))
        # get the temporal range of the start point of the mask window
        segment = list(range(int(win_train - mask_rate * win_train)))
        # segment = list(range(int(win_train)))
        # get the randomlized value of start point of the mask window
        r_m = sample(segment, 1)[0]
        # get training samples of batchsize trials
        for i in range(batchsize):
            k = sample(train_list, 1)[0]
            m = sample(target_list, 1)[0]
            # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
            time_start = random.randint(125 + 35, int(1250 + 125 + 35 - win_train))
            time_end = time_start + win_train
            # get three sub-inputs
            # x_11 = train_data1[k][m][:,time_start:time_end]
            # x_21 = np.reshape(x_11,(channel, win_train, 1))
            # x_train1[i]=x_21

            # x_12 = train_data2[k][m][:,time_start:time_end]
            # x_22 = np.reshape(x_12,(channel, win_train, 1))
            # x_train2[i]=x_22

            # x_13 = train_data3[k][m][:,time_start:time_end]
            # x_23 = np.reshape(x_13,(channel, win_train, 1))
            # x_train3[i]=x_23

            # 2024/08/03/18/20
            x_11 = train_data1[k, m, :, time_start:time_end]
            # x_11 = train_data1[k][m][:, time_start:time_end]
            x_21 = np.reshape(x_11, (channel, win_train, 1))
            x_train1[i] = x_21

            x_12 = train_data2[k, m, :, time_start:time_end]
            # x_12 = train_data2[k][m][:, time_start:time_end]
            x_22 = np.reshape(x_12, (channel, win_train, 1))
            x_train2[i] = x_22

            x_13 = train_data3[k, m, :, time_start:time_end]
            # x_13 = train_data3[k][m][:, time_start:time_end]
            x_23 = np.reshape(x_13, (channel, win_train, 1))
            x_train3[i] = x_23

            y_train[i] = to_categorical(m, num_classes=40, dtype='float32')

        x_train1 = np.reshape(x_train1, (batchsize, channel, win_train, 1))
        x_train2 = np.reshape(x_train2, (batchsize, channel, win_train, 1))
        x_train3 = np.reshape(x_train3, (batchsize, channel, win_train, 1))

        # concatenate the four sub-input into one input to make it can be as the input of the CNN-Former's network
        x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)

        # #appling the EEG-ME to the EEG samples
        # insert_length = int(mask_rate * 250)
        # data_to_insert = np.zeros((batchsize, channel, insert_length, 3))

        x_train[:, :, r_m:int(r_m + mask_rate * win_train), :] = 0
        # x_train = np.concatenate((x_train[:, :, :r_m, :], data_to_insert, x_train[:, :, r_m:, :]), axis=2)

        # 2024/08/02/
        # num_segments= 1
        # insert_length = int(mask_rate * win_train / num_segments)
        # # print("insert_length:",insert_length)
        # data_to_insert = np.zeros((batchsize, channel, insert_length, 3))
        # # print("data_to_insert:",data_to_insert.shape)
        # for _ in range(num_segments):
        #     r_m = random.randint(0, win_train)
        #     x_train = np.concatenate((x_train[:, :, :r_m, :], data_to_insert, x_train[:, :, r_m:, :]), axis=2)
        # x_train[:, :, r_m:r_m + insert_length, :] = data_to_insert
        # y_train = np.reshape(y_train, (batchsize, 40))

        y_train = np.reshape(y_train, (batchsize, 40))

        yield x_train, y_train



