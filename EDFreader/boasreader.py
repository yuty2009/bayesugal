# -*- coding: utf-8 -*-

import os
import pyedflib
import sys
import torch
import tqdm
import pandas as pd
from torch.utils.data import Dataset
from EDFreader.signalproc import *
from pre.sleepstage import stage_dict
from scipy import interpolate


# translate epoch data (n, t, c) into grayscale images (n, 1, t, c)
class TransformEpoch(object):
    def __call__(self, epoch):
        epoch = torch.Tensor(epoch)
        return torch.unsqueeze(epoch, dim=0)


class TransformSTFT(object):
    """ Translate epoch data (t, c) into grayscale images (1, N, T, c)
    # N is the number of frequencies where STFT is applied and
    # T is the total number of frames used.
    """
    def __call__(self, epoch):
        epoch = torch.FloatTensor(epoch)
        epoch_spectra = []
        channels = epoch.size(-1)
        for i in range(channels):
            spectra = torch.stft(
                epoch[..., i],
                n_fft=256,
                hop_length=100,
                win_length=200,
                window=torch.hamming_window(200),
                center=False,
                onesided=True,
                return_complex=False,
            )
            spectra_real = spectra[..., 0]
            spectra_imag = spectra[..., 1]
            spectra_magn = torch.abs(torch.sqrt(torch.pow(spectra_real,2)+torch.pow(spectra_imag,2)))
            spectra_magn = 20*torch.log10(spectra_magn)
            epoch_spectra.append(spectra_magn.unsqueeze(dim=-1))
        epoch_spectra = torch.cat(epoch_spectra, dim=-1)
        if channels <= 1:
            epoch_spectra = epoch_spectra.squeeze(-1)
        return epoch_spectra.unsqueeze(dim=0)


class EEGDataset(Dataset):
    def __init__(self, epochs, labels, transforms=None):
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.labels = torch.Tensor(labels).long()

    def __getitem__(self, idx):
        return self.epochs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class SeqEEGDataset(Dataset):
    def __init__(self, epochs, labels, seqlen, transforms=None):
        if transforms == None:
            self.epochs = torch.Tensor(epochs)
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
            self.epochs = torch.stack(self.epochs)

        self.labels = torch.Tensor(labels).long()
        self.seqlen = seqlen
        assert self.seqlen <= len(self), "seqlen is too large"

    def __getitem__(self, idx):
        epoch_seq = torch.zeros(
            (self.seqlen,)+self.epochs.shape[1:], 
            dtype=self.epochs.dtype,
            device=self.epochs.device
            )
        idx1 = idx + 1
        if idx1 < self.seqlen:
            epoch_seq[-idx1:] = self.epochs[:idx1]
        else:
            epoch_seq = self.epochs[idx1-self.seqlen:idx1]
        return epoch_seq, self.labels[idx]

    def __len__(self):
        return len(self.labels) 


_available_dataset = [
    'sleep-edf-v1',
    'sleep-edf-ex',
    ]

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def resample(signal, signal_frequency, target_frequency):
    resampling_ratio = signal_frequency / target_frequency
    x_base = np.arange(0, len(signal))

    interpolator = interpolate.interp1d(x_base, signal, axis=0, bounds_error=False, fill_value='extrapolate', )

    x_interp = np.arange(0, len(signal), resampling_ratio)

    signal_duration = signal.shape[0] / signal_frequency
    resampled_length = round(signal_duration * target_frequency)
    resampled_signal = interpolator(x_interp)
    if len(resampled_signal) < resampled_length:
        padding = np.zeros((resampled_length - len(resampled_signal), signal.shape[-1]))
        resampled_signal = np.concatenate([resampled_signal, padding])

    return resampled_signal


def load_eegdata(setname, datapath, subject):
    assert setname in _available_dataset, 'Unknown dataset name ' + setname
    if setname == 'sleepedf':
        filepath = os.path.join(datapath, subject+'.rec')
        labelpath = os.path.join(datapath, subject+'.hyp')
        data, target = load_eegdata_sleepedfx(filepath, labelpath)
    if setname == 'sleepedfx':
        filepath = os.path.join(datapath, subject+'-PSG.edf')
        labelpath = os.path.join(datapath, subject+'-Hypnogram.edf')
        data, target = load_eegdata_sleepedfx(filepath, labelpath)
    return data, target


def load_eegdata_sleepedf(rec_fname, hyp_fname):
    data = []
    target = []
    return data, target


def load_eegdata_boas(psg_fname, ann_fname, select_ch=['HB_1', 'HB_2']):
    """
    https://github.com/akaraspt/tinysleepnet
    """

    W, N1, N2, N3, REM = 0, 0, 0, 0, 0
    psg_f = pyedflib.EdfReader(psg_fname)
    label_f = pd.read_csv(ann_fname, sep='\t')

    # assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
    start_datetime = psg_f.getStartdatetime()

    file_duration = psg_f.getFileDuration()
    epoch_duration = psg_f.datarecord_duration

    # Extract signal from the selected channel
    ch_names = psg_f.getSignalLabels()
    ch_samples = psg_f.getNSamples()
    if len(select_ch) ==2:
        select_ch_idx0, select_ch_idx1 = -1, -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == select_ch[0]:
                select_ch_idx0 = s
                break
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == select_ch[1]:
                select_ch_idx1 = s
                break
        if select_ch_idx0 == -1:
            raise Exception("Channel0 not found.")
        if select_ch_idx1 == -1:
            raise Exception("Channel1 not found.")
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx0)
        n_epoch_samples = int(epoch_duration * sampling_rate)
        # signals = psg_f.readSignal(select_ch_idx).reshape(-1, n_epoch_samples)
        signals0 = psg_f.readSignal(select_ch_idx0).reshape(-1, 256)
        signals1 = psg_f.readSignal(select_ch_idx1).reshape(-1, 256)
        signals = signals0 - signals1
    if len(select_ch) == 1:
        select_ch_idx0 = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == select_ch[0]:
                select_ch_idx0 = s
                break
        if select_ch_idx0 == -1:
            raise Exception("Channel0 not found.")

        sampling_rate = psg_f.getSampleFrequency(select_ch_idx0)
        n_epoch_samples = int(epoch_duration * sampling_rate)
        signals = psg_f.readSignal(select_ch_idx0).reshape(-1, 256)
    signals = resample(signals.T, sampling_rate, 100).T
    labels = label_f.iloc[:, -2].values
    ai_labels = label_f.iloc[:, -1].values
    total = len(labels)
    diff = np.sum(labels != ai_labels)
    # Sanity check
    n_epochs = psg_f.datarecords_in_file

    signals_list, labels_list = [], []
    for i in range(len(labels)):
        if 0 <= labels[i] <= 4:
            labels_list.append(labels[i])
            signals_list.append(signals[i*30:(i+1)*30].reshape(3000,))
            if labels[i] == 0: W += 1
            if labels[i] == 1: N1 += 1
            if labels[i] == 2: N2 += 1
            if labels[i] == 3: N3 += 1
            if labels[i] == 4: REM += 1

    x = np.vstack(signals_list).astype(np.float32)
    y = np.hstack(labels_list).astype(np.int32)

    # # Generate labels from onset and duration annotation
    # labels = []
    # total_duration = 0
    # for a in range(len(ann_stages)):
    #     onset_sec = int(ann_onsets[a])
    #     duration_sec = int(ann_durations[a])
    #     ann_str = "".join(ann_stages[a])
    #
    #     # Sanity check
    #     assert onset_sec == total_duration
    #
    #     # Get label value
    #     label = ann2label[ann_str]
    #
    #     # Compute # of epoch for this stage
    #     if duration_sec % epoch_duration != 0:
    #         raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
    #     duration_epoch = int(duration_sec / epoch_duration)
    #
    #     # Generate sleep stage labels
    #     label_epoch = np.ones(duration_epoch, dtype=int) * label
    #     labels.append(label_epoch)
    #
    #     total_duration += duration_sec
    #
    # labels = np.hstack(labels)
    #
    # # Remove annotations that are longer than the recorded signals
    # labels = labels[:len(signals)]
    #
    # # Get epochs and their corresponding labels
    # x = signals.astype(np.float32)
    # y = labels.astype(np.int32)
    #
    # # Select only sleep periods
    # w_edge_mins = 30
    # nw_idx = np.where(y != stage_dict["W"])[0]
    # start_idx = nw_idx[0] - (w_edge_mins * 2)
    # end_idx = nw_idx[-1] + (w_edge_mins * 2)
    # if start_idx < 0:
    #     start_idx = 0
    # if end_idx >= len(y):
    #     end_idx = len(y) - 1
    # select_idx = np.arange(start_idx, end_idx+1)
    # x = x[select_idx]
    # y = y[select_idx]
    #
    # # Remove movement and unknown
    # move_idx = np.where(y == stage_dict["MOVE"])[0]
    # unk_idx = np.where(y == stage_dict["UNK"])[0]
    # if len(move_idx) > 0 or len(unk_idx) > 0:
    #     remove_idx = np.union1d(move_idx, unk_idx)
    #     select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
    #     x = x[select_idx]
    #     y = y[select_idx]

    # Save
    data_dict = {
        "x": x, 
        "y": y, 
        "fs": sampling_rate,
        "ch_label": select_ch,
        "start_datetime": start_datetime,
        "file_duration": file_duration,
        "epoch_duration": epoch_duration,
        "n_all_epochs": n_epochs,
        "n_epochs": len(x),
    }

    return data_dict, W, N1, N2, N3, REM, total, diff


def extract_rawfeature(data, target, filter, sampleseg, chanset, standardize=True):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = sample_end - sample_begin
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    features = np.zeros([num_trials, num_samples_used, num_channel_used])
    for i in range(num_trials):
        signal_epoch = data[i]
        signal_filtered = signal_epoch
        for j in range(num_channels):
            signal_filtered[:, j] = signal.lfilter(filter[0], filter[1], signal_filtered[:, j])
            # signal_filtered[:, j] = signal.filtfilt(filter[0], filter[1], signal_filtered[:, j])
        if standardize:
            # init_block_size=1000 this param setting has a big impact on the result
            signal_filtered = exponential_running_standardize(signal_filtered, init_block_size=1000)
        features[i] = signal_filtered[sample_begin:sample_end, chanset]

    return features, labels


def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        # print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # Reshape the data to match the input of the model - conv2d
        # tmp_data = np.squeeze(tmp_data)
        # tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
        
        # # Reshape the data to match the input of the model - conv1d
        tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data.append(tmp_data)
        labels.append(tmp_labels)

    return data, labels


def load_subdata_preprocessed(datapath, subject):
    npz_f = os.path.join(datapath, subject+'.npz')
    data, labels, fs = load_npz_file(npz_f)
    return data, labels


def load_dataset_preprocessed(datapath, n_subjects=None):
    allfiles = os.listdir(datapath)
    npzfiles = []
    data_bar = tqdm.tqdm(total=len(allfiles), file=sys.stdout)
    for f in allfiles:
        if ".npz" in f:
            npzfiles.append(os.path.join(datapath, f))
        data_bar.update()
    data_bar.close()
    npzfiles.sort()
    if n_subjects is not None:
        npzfiles = npzfiles[:n_subjects]
    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]
    data, labels = load_npz_list_files(npzfiles)
    return data, labels, subjects


if __name__ == '__main__':

    import os
    import glob

    datapath = 'D:/dataset/BOAS/'
    savepath = 'D:/dataset/BOAS/processed/'
    os.makedirs(savepath, exist_ok=True)

    subs = glob.glob(os.path.join(datapath, "sub-*"))

    psg_fnames, ann_fnames = [], []
    for i in range(len(subs)):
        psg_fnames.append(glob.glob(os.path.join(subs[i] + '/eeg/', "*psg_eeg.edf"))[0])
        ann_fnames.append(glob.glob(os.path.join(subs[i] + '/eeg/', "*psg_events.tsv"))[0])
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    totalW, totalN1, totalN2, totalN3, totalREM, totalT, totalD = 0, 0, 0, 0, 0, 0, 0
    for i in range(len(psg_fnames)):

        subject = os.path.basename(psg_fnames[i])

        print('Load and extract continuous EEG into epochs for subject： '+subject)
        data_dict, curW, curN1, curN2, curN3, curREM, total, diff = load_eegdata_boas(psg_fnames[i], ann_fnames[i], select_ch=['HB_1'])

        totalW += curW
        totalN1 += curN1
        totalN2 += curN2
        totalN3 += curN3
        totalREM += curREM
        totalT += total
        totalD += diff
        np.savez(savepath + 'sub' + str(i) + '.npz', **data_dict)
    print(totalW)
    print(totalN1)
    print(totalN2)
    print(totalN3)
    print(totalREM)
    print(totalT)
    print(totalD)