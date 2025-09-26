import numpy as np
import torch
import scipy.io as scio
from os import path
from scipy import signal

path_Extracted = 'D:/dataset/isruc/Data of Subgroup_3/MatData/'
path_RawData   = 'D:/dataset/isruc/Data of Subgroup_3/RawData/'
path_output    = 'D:/dataset/isruc/Subgroup_3_output_processed/'
channels = ['C3_A2']


def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    output = []
    output.append(np.array(label[:-ignore]))
    return np.concatenate(output, axis=0)


'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        Fold_label: [k-fold] list, each element is [N,C]
        Fold_len:   [k-fold] list
'''

fold_label = []
fold_psg = []
fold_len = []

for sub in range(1, 11):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_Extracted, sub, channels)
    psg = psg.reshape(psg.shape[0], -1)
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label==5] = 4  # make 4 correspond to REM

    data_dict1 = {
        "x": psg,
        "y": label,
        "fs": 200,
        "ch_label": 'C3_A2',
        "n_epochs": len(label)
    }
    np.savez(path_output + str(sub) + '.npz', **data_dict1)

print('Preprocess over.')
