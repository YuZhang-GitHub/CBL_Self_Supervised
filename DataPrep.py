# the loaded data type in MATLAB is struct with 2 fields: channels (100, 1, 1184923) and userLoc (1184923, 3)
# channels: (#ant, #sub, #user), userLoc: (#user, 3)
# And also note that each element of channels is complex number.
import numpy as np
import h5py as h5


def dataPrep(inputName=None, valPerc=0.3, save_shuffled_idx=False):
    with h5.File(inputName, 'r') as f:
        fields = [k for k in f.keys()]
        nested = [k for k in f[fields[1]].keys()]
        deepMIMO_data = f[fields[1]]
        data_channels = np.squeeze(np.array(deepMIMO_data[nested[0]]))
        # shape: (#users, #ant), in #ant dim, it is a tuple with real and imag parts of original data (real, imag)
        # data_userLoc = np.array(deepMIMO_data[nested[1]]) # shape: (3, #users)
        decoup = data_channels.view(np.float32).reshape(data_channels.shape + (2,))
        # shape: (#users, #ant, 2), decoup[0,0,0]=real, decoup[0,0,1]=imag
        X_real = decoup[:, :, 0]
        X_imag = decoup[:, :, 1]
        # data_channels = concat(data_channels) # shape: (#users, #sub, #ant), in #ant dim, it is a complex number
        X = np.concatenate((X_real, X_imag), axis=-1)

    print('Creating training and validation inputs')
    numTrain = np.floor((1 - valPerc) * X.shape[0]).astype('int')
    numVal = np.floor(valPerc * X.shape[0]).astype('int')
    print('Size of training dataset: ' + str(numTrain) + '\n' + 'Size of validation dataset: ' + str(numVal))
    shuffled_idx = np.random.permutation(X.shape[0])
    if save_shuffled_idx:
        np.save('shuffled_ind', shuffled_idx)
    X_shuffled = X[shuffled_idx, :]
    train_inp = X_shuffled[0:numTrain, :]
    val_inp = X_shuffled[numTrain:, :]

    return train_inp, val_inp


def load_data(load_file, tr_load_perc=1):
    train_inp = np.load(load_file + 'trainning_clean_set.npy')  # shape: (numTrain, 2*#ant)
    val_inp = np.load(load_file + 'validation_clean_set.npy')
    total_train = train_inp.shape[0]
    return train_inp[:tr_load_perc * total_train, :], val_inp


# Debugging ONLY or generating datasets and save
# inputFile = 'F:\CodebookLearning_Clean_Generation\CBL_O1_60_BS3_60GHz_1Path_Corrupted_norm.mat'
# train_inp, val_inp = dataPrep(inputName=inputFile, save_shuffled_idx=True)
# np.save('trainning_set', train_inp)
# np.save('validation_set', val_inp)
