from tools import *
from pre.utils import *
from EDFreader.sleepreader import *
from model import BayesianConvNet
from sklearn.model_selection import KFold


def main():
    """
    ----------------------------------------------- hyperparameter----------------------------------------------------
    """
    device_id  = 0
    n_seqlen   = 20
    n_classes  = 5
    n_epochs   = 200
    lr         = 12e-4
    batch_size = 1024
    kl_weight  = 2e-8
    save       = True
    """
    ----------------------------------------------- Chose Model Type--------------------------------------------------
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
    bayesian   = model_data[model_type]['bayesian' ]
    mc_drop    = model_data[model_type]['mc_drop'  ]
    model_num  = model_data[model_type]['model_num']
    """
    ------------------------- Chose dataset: 0 - edf20, 1 - edf78, 2- MASS, 3- physio2018 -----------------------------
    """
    dataset = 0
    dataset_paths = {
        0: {'path': 'sleep-edf-20',  'n_subjects': 39 , 'folds': 10},
        1: {'path': 'sleep-edf-78',  'n_subjects': 153, 'folds': 10},
        2: {'path': 'MASS',          'n_subjects': 200, 'folds': 10},
        3: {'path': 'physionet2018', 'n_subjects': 994, 'folds': 5 },
        4: {'path': 'BOAS_HB2',      'n_subjects': 124, 'folds': 10},
        5: {'path': 'BOAS_PSG',      'n_subjects': 124, 'folds': 10}
    }
    """
    -----------------------------------------------data loading -------------------------------------------------------
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
    n_timepoints = data[0].shape[-2]
    tf_epoch = TransformEpoch()
    torch.manual_seed(42)
    """
    --------------------------------------- fold splitting and training ------------------------------------------------
    """
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    splits_train, splits_test = [], []
    for (a, b) in kfold.split(np.arange(n_subjects)):
        splits_train.append(a)
        splits_test.append(b)

    for fold in range(0, int(np.ceil(folds/2))):
        idx_train, idx_test = splits_train[fold], splits_test[fold]
        trainset = [SeqEEGDataset(data[i], labels[i], n_seqlen, tf_epoch) for i in idx_train]
        train_dataset = torch.utils.data.ConcatDataset(trainset)
        train_dataloader = MultiEpochsDataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=8, pin_memory=True)
        model = []
        optimizer = []
        for i in range(model_num):
            model.append(BayesianConvNet(n_timepoints, n_seqlen, n_classes, bayesian, mc_drop, device).to(device))
        parameters = sum(p.numel() for p in model[0].parameters() if p.requires_grad)
        print("num of trainable parameters per model:", parameters)
        parameters *= model_num
        print("total num of trainable parameters:", parameters)
        print()

        for i in range(model_num):
            optimizer.append(torch.optim.AdamW(model[i].parameters(), lr, betas=(0.9, 0.95)))
        start = time.time()
        for epoch in range(n_epochs):
            for i in range(model_num):
                adjust_learning_rate(optimizer[i], epoch, n_epochs, lr)
            train_epoch(train_dataloader, model_num, model, optimizer, device, kl_weight)
            print(f"Fold: {fold}, Epoch: {epoch}| Elapsed time = {as_minutes(time.time()-start)}")
            print()
        if save:  # saving model
            torch.save(model, savepath + str(fold))



if __name__ == '__main__':
    main()
