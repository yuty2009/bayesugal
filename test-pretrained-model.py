from tools import *
from torch.utils.data import random_split
from pre.utils import *
from EDFreader.sleepreader import *
from sklearn.model_selection import KFold


def main():
    device_id  = 0
    n_seqlen   = 20
    batch_size = 1024
    save       = True
    """
    >>>> Chose Model Type
    """
    model_type = 'remove_none'
    model_data = {
        'remove_vote' : {'bayesian': True , 'mc_drop': True , 'model_num': 1},
        'remove_mc'   : {'bayesian': True , 'mc_drop': False, 'model_num': 2},
        'remove_bayes': {'bayesian': False, 'mc_drop': True , 'model_num': 2},
        'remove_none' : {'bayesian': True , 'mc_drop': True , 'model_num': 2},
        'baseline'    : {'bayesian': False, 'mc_drop': False, 'model_num': 1},
        '3_model_vote': {'bayesian': True , 'mc_drop': True , 'model_num': 3},
        '4_model_vote': {'bayesian': True , 'mc_drop': True , 'model_num': 4},
        '5_model_vote': {'bayesian': True , 'mc_drop': True , 'model_num': 5}
    }
    bayesian  = model_data[model_type]['bayesian' ]
    mc_drop   = model_data[model_type]['mc_drop'  ]
    model_num = model_data[model_type]['model_num']
    """
    >>>> Chose dataset: 0 - edf20, 1 - edf78, 2- MASS, 3- physio2018 
    """
    dataset = 0
    dataset_paths = {
        0: {'path': 'sleep-edf-20' , 'n_subjects': 39 , 'folds': 10},
        1: {'path': 'sleep-edf-78' , 'n_subjects': 153, 'folds': 10},
        2: {'path': 'MASS'         , 'n_subjects': 200, 'folds': 10},
        3: {'path': 'physionet2018', 'n_subjects': 994, 'folds': 5 },
        4: {'path': 'BOAS_HB2',          'n_subjects': 124, 'folds': 10},
        5: {'path': 'BOAS_PSG',      'n_subjects': 124, 'folds': 10}
    }
    """
    -----------------------------------------------------------------------------------------------
    """
    torch.backends.cudnn.benchmark = True
    device_id = "cuda:" + str(device_id)
    device = torch.device(device_id)
    datapath = 'H:/dataset/' + dataset_paths[dataset]['path'] + '/processed/'
    savepath = ('E:/result/' + dataset_paths[dataset]['path'] + '_bayesian' + str(int(bayesian))
                + '_mc' + str(int(mc_drop)) + '_num' + str(int(model_num)) + '_')
    folds = dataset_paths[dataset]['folds']
    n_subjects = dataset_paths[dataset]['n_subjects']
    data, labels, subjects = load_dataset_preprocessed(datapath, n_subjects=n_subjects)
    print('Data for %d subjects has been loaded' % len(data))
    n_subjects = len(data)
    tf_epoch   = TransformEpoch()
    torch.manual_seed(42)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    splits_train, splits_test = [], []
    for (a, b) in kfold.split(np.arange(n_subjects)):
        splits_train.append(a)
        splits_test.append(b)

    true_num = np.zeros(shape=folds, dtype=float)
    test_num = np.zeros(shape=folds, dtype=float)
    tp, tn, fp, fn, a, b = (torch.zeros((folds, 5)), torch.zeros((folds, 5)), torch.zeros((folds, 5)),
                            torch.zeros((folds, 5)), torch.zeros((folds, 5)), torch.zeros((folds, 5)))
    start = time.time()
    for fold in range(folds):
        idx_train, idx_test = splits_train[fold], splits_test[fold]
        testset  = [SeqEEGDataset(data[i], labels[i], n_seqlen, tf_epoch) for i in idx_test ]
        test_dataset  = torch.utils.data.ConcatDataset(testset )
        test_dataloader  = MultiEpochsDataLoader(test_dataset , batch_size=batch_size,
                                                 shuffle=True, num_workers=4, pin_memory=True)
        model = torch.load(savepath + str(fold), map_location=device)
        true_num[fold], test_num[fold], tp[fold], fp[fold], tn[fold], fn[fold], a[fold], b[fold]\
            = test(test_dataloader, model, model_num, device)
        print(f"Fold: {fold}| Elapsed time = {as_minutes(time.time() - start)}")
        print()

    test_num = sum(test_num)/folds
    true_num = sum(true_num) / folds
    tp, tn, fp, fn = torch.sum(tp, dim=0), torch.sum(tn, dim=0), torch.sum(fp, dim=0), torch.sum(fn, dim=0)
    a, b = torch.sum(a, dim=0), torch.sum(b, dim=0)

    pr, re = torch.zeros(5), torch.zeros(5)
    f1 = torch.zeros(5)
    for y in range(5):
        pr[y] = torch.sum(tp[y]) / (torch.sum(tp[y]) + torch.sum(fp[y]))
        re[y] = torch.sum(tp[y]) / (torch.sum(tp[y]) + torch.sum(fn[y]))
    for y in range(5):
        f1[y] = 2 * pr[y] * re[y] / (pr[y] + re[y])
    pe = torch.sum(a * b) / test_num / test_num / folds / folds
    mf1 = 0.4 * (torch.sum(pr * re / (pr + re))) * 100
    k = 1 - (1 - true_num / test_num / 100) / (1 - pe)
    print('acc:  ' + str(true_num/test_num))
    print('mf1:  ' + str(mf1.item()))
    print('k:    ' + str(k.item()))
    print('W:    ' + str(f1[0].item()*100))
    print('N1:   ' + str(f1[1].item()*100))
    print('N2:   ' + str(f1[2].item()*100))
    print('N3:   ' + str(f1[3].item()*100))
    print('REM:  ' + str(f1[4].item()*100))


if __name__ == '__main__':
    main()
