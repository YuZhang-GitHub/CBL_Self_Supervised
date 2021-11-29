import numpy as np
import scipy.io as scio
from DataPrep import load_data
from Model import Model
import sys
from utils import bf_gain_cal, train_shuffle

num_ant = 64
num_of_beams = [64]
load_file = './'

batch_size = 500
epoch_schedule = [2, 3]
lr_schedule = [0.1, 0.01]

if len(epoch_schedule) != len(lr_schedule):
    print('Reset epoch schedule and learning rate schedule.')
    sys.exit()

# Input_data loading and preparation
train_inp, val_inp = load_data(load_file)

num_train_batch = np.floor(train_inp.shape[0] / batch_size).astype('int')
num_val_batch = np.floor(val_inp.shape[0] / batch_size).astype('int')
limit = num_train_batch * batch_size
train_data = train_inp[0:limit, :]
limit = num_val_batch * batch_size
val_data = val_inp[0:limit, :]

input_size = train_data.shape[1]  # 2 * num_ant
for num_beams in num_of_beams:
    print(str(num_beams) + '-beams Codebook is being generated...')

    # Model:
    net = Model(num_beams, num_ant, batch_size, mode='recon', accum=True)

    val_gain_iter = [bf_gain_cal(net.codebook, val_inp, num_ant)]  # This is for plotting
    print('Initial codebook performance (val gain): %f.' % val_gain_iter[0])

    for val_idx in range(val_data.shape[0]):
        net.forward(val_data[val_idx, :], val_mode=True, val_size=val_data.shape[0])
    val_loss = net.Loss.loss_val
    print('Initial codebook performance (val loss): %f.' % val_loss)
    val_loss_iter = [val_loss]

    for tr_idx in range(train_data.shape[0]):
        net.forward(train_data[tr_idx, :], val_mode=True, val_size=train_data.shape[0])
    tr_loss = net.Loss.loss_val
    print('Initial codebook performance (tr loss): %f.' % tr_loss)
    tr_loss_iter = [tr_loss]

    # Training:
    for sch_idx in range(len(epoch_schedule)):
        for epoch_idx in range(epoch_schedule[sch_idx]):
            train_data = train_shuffle(train_data)
            for batch_idx in range(num_train_batch):
                grad = np.zeros([num_beams, num_ant])
                for ch_idx in range(batch_size):
                    channel = train_data[batch_idx * batch_size + ch_idx, :]
                    net.forward(channel)
                    net.backward()
                loss = net.Loss.loss
                tr_loss_iter.append(loss)
                net.codebook = net.update(lr=lr_schedule[sch_idx])

                # Validation
                for val_idx in range(val_data.shape[0]):
                    net.forward(val_data[val_idx, :], val_mode=True, val_size=val_data.shape[0])

                val_gain = bf_gain_cal(net.codebook, val_inp, num_ant)
                val_gain_iter.append(val_gain)
                val_loss = net.Loss.loss_val
                val_loss_iter.append(val_loss)

                print('beam: %d, schedule: %d (%d), epoch: %d (%d), batch: %d, loss: %f, avg gain: %f, val loss: %f.' %
                      (num_beams, sch_idx + 1, len(epoch_schedule), epoch_idx + 1, epoch_schedule[sch_idx],
                       batch_idx + 1, loss, val_gain, val_loss))

                if net.codebook.dtype != 'float64':
                    ValueError('Bad thing happens!')

    # Output:
    theta = np.transpose(net.codebook)  # To MATLAB format: (# fo antennas, # of beams)
    print(theta.shape)
    name_of_file = 'theta_self_sup_' + str(num_beams) + 'beams.mat'
    scio.savemat(name_of_file,
                 {'train_inp': train_data,
                  'val_inp': val_data,
                  'codebook': theta,
                  'val_gain_iter': val_gain_iter,
                  'val_loss_iter': val_loss_iter,
                  'tr_loss_iter': tr_loss_iter})
