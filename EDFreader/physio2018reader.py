import numpy as np
import pandas as pd
import gc
import os
import glob
import scipy


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res


def init():
    # Create the 'models' subdirectory and delete any existing model files
    try:
        os.mkdir('models')
    except OSError:
        pass
    for f in glob.glob('models/*_model.pkl'):
        os.remove(f)


def get_files(rootDir = 'D:/dataset/physionet2018/challenge-2018'):
    header_loc, arousal_loc, signal_loc, is_training = [], [], [], []
    for dirName, subdirList, fileList in os.walk(rootDir, followlinks=True):
        if (dirName != 'D:/dataset/physionet2018/challenge-2018'
                and dirName != 'D:/dataset/physionet2018/challenge-2018/test'
                and dirName != 'D:/dataset/physionet2018/challenge-2018/training'):
            if dirName.startswith('D:/dataset/physionet2018/challenge-2018\\training\\'):
                is_training.append(True)

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '/' + fname)
                    if '-arousal.mat' in fname:
                        arousal_loc.append(dirName + '/' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '/' + fname)

            # elif dirName.startswith('./test/'):
            #     is_training.append(False)
            #     arousal_loc.append('')
            #
            #     for fname in fileList:
            #         if '.hea' in fname:
            #             header_loc.append(dirName + '/' + fname)
            #         if 'mat' in fname and 'arousal' not in fname:
            #             signal_loc.append(dirName + '/' + fname)

    # combine into a data frame
    data_locations = {'header':      header_loc,
                      'arousal':     arousal_loc,
                      'signal':      signal_loc,
                      'is_training': is_training
                      }

    # Convert to a data-frame
    df = pd.DataFrame(data=data_locations)

    # Split the data frame into training and testing sets.
    tr_ind = list(find(df.is_training.values))
    # te_ind = list(find(df.is_training.values == False))

    training_files = df.loc[tr_ind, :]
    # testing_files  = df.loc[te_ind, :]

    return training_files


def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples


def import_arousals(file_name):
    import h5py
    import numpy
    f = h5py.File(file_name, 'r')

    wake = numpy.array(f['data']['sleep_stages']['wake'])
    nonrem1 = numpy.array(f['data']['sleep_stages']['nonrem1'])
    nonrem2 = numpy.array(f['data']['sleep_stages']['nonrem2'])
    nonrem3 = numpy.array(f['data']['sleep_stages']['nonrem3'])
    rem = numpy.array(f['data']['sleep_stages']['rem'])
    undefined = numpy.array(f['data']['sleep_stages']['undefined'])
    arousals = nonrem1 + nonrem2*2 + nonrem3*3 + rem*4 + undefined*5
    return arousals


def import_signals(file_name):
    return np.transpose(scipy.io.loadmat(file_name)['val'])


def get_subject_data(arousal_file, signal_file, signal_names):
    this_arousal   = import_arousals(arousal_file).T
    this_signal    = import_signals(signal_file)
    this_data      = np.append(this_signal, this_arousal, axis=1)
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data


def preprocess_record(record_name, time=30):
    header_file = record_name + '.hea'
    signal_file = record_name + '.mat'
    arousal_file = record_name + '-arousal.mat'

    # Get the signal names from the header file
    signal_names, Fs, n_samples = import_signal_names(header_file)
    signal_names = list(np.append(signal_names, 'arousals'))

    # Convert this subject's data into a pandas dataframe
    this_data = get_subject_data(arousal_file, signal_file, signal_names)

    # ----------------------------------------------------------------------
    # Generate the Features for the classificaition model - variance of SaO2
    # ----------------------------------------------------------------------

    # For the baseline, let's only look at how SaO2 might predict arousals

    signal = this_data.get(['C3-M2']).values
    arousals = this_data.get(['arousals']).values
    step        = Fs * time

    sample = []
    label = []
    i = 0
    while (i * step + step - 1) <= n_samples:
        if arousals[i * step + step//2] != 5:
            cur_signal = np.transpose(signal[(i * step):(i * step + step)])
            sample.append(scipy.signal.resample(cur_signal, 3000, axis=-1))
            label.append(arousals[i * step + step//2])
        i += 1
    sample = np.concatenate(sample, axis=0)
    label = np.concatenate(label, axis=0)

    return sample, label


if __name__ == "__main__":
    init()
    rootDir = 'D:/dataset/physionet2018/challenge-2018'
    savepath = 'D:/dataset/physionet2018/processed2'

    # Generate a data frame that points to the challenge files
    tr_files = get_files(rootDir)

    for i in range(0, np.size(tr_files, 0)):
        gc.collect()
        print('Preprocessing training subject: %d/%d'
              % (i + 1, np.size(tr_files, 0)))
        record_name = tr_files.header.values[i][:-4]
        print(record_name)
        x, y = preprocess_record(record_name, time=30)
        data_dict = {
            "x": x,
            "y": y,
            "fs": 200,
            "ch_label": 'C3-M2'
        }
        np.savez(savepath + '/' + record_name[-9:] + '.npz', **data_dict)