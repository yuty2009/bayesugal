from tools import *
from pre.utils import *
from EDFreader.sleepreader import *
from model import BayesianConvNet
from sklearn.model_selection import KFold


def main():
    """
    ----------------------------------------------- hyperparameter----------------------------------------------------
    """
    device_id = 0
    n_seqlen = 20
    n_classes = 5
    n_epochs = 200
    lr = 12e-4
    batch_size = 1024
    kl_weight = 2e-8
    """
    ----------------------------------------------- Chose Model Type--------------------------------------------------
    """
    model_type = 'remove_none'
    model_data = {
        'remove_vote': {'bayesian': True, 'mc_drop': True, 'model_num': 1},
        'remove_mc': {'bayesian': True, 'mc_drop': False, 'model_num': 2},
        'remove_bayes': {'bayesian': False, 'mc_drop': True, 'model_num': 2},
        'remove_none': {'bayesian': True, 'mc_drop': True, 'model_num': 2},
        'baseline': {'bayesian': False, 'mc_drop': False, 'model_num': 1},
        '3_model_vote': {'bayesian': True, 'mc_drop': True, 'model_num': 3},
        '4_model_vote': {'bayesian': True, 'mc_drop': True, 'model_num': 4},
        '5_model_vote': {'bayesian': True, 'mc_drop': True, 'model_num': 5}
    }
    bayesian = model_data[model_type]['bayesian']
    mc_drop = model_data[model_type]['mc_drop']
    model_num = model_data[model_type]['model_num']
    """
    ------------------------- Chose dataset: 0 - edf20, 1 - edf78, 2- MASS, 3- physio2018 -----------------------------
    """
    dataset = 0
    dataset_paths = {
        0: {'path': 'sleep-edf-20', 'n_subjects': 39, 'folds': 10},
        1: {'path': 'sleep-edf-78', 'n_subjects': 153, 'folds': 10},
        2: {'path': 'MASS', 'n_subjects': 200, 'folds': 10},
        3: {'path': 'physionet2018', 'n_subjects': 994, 'folds': 5}
    }
    """
    -----------------------------------------------data loading -------------------------------------------------------
    """
    torch.backends.cudnn.benchmark = True
    device_id = "cuda:" + str(device_id)
    device = torch.device(device_id)
    datapath = 'D:/dataset/' + dataset_paths[dataset]['path'] + '/processed/'
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
    ---------------------------------------- evaluation metrics definition----------------------------------------------
    """
    true_num1 = np.zeros(shape=folds, dtype=float)
    test_num1 = np.zeros(shape=folds, dtype=float)
    tp1, tn1, fp1, fn1, a1, b1 = (torch.zeros((folds, 5)), torch.zeros((folds, 5)), torch.zeros((folds, 5)),
                                  torch.zeros((folds, 5)), torch.zeros((folds, 5)), torch.zeros((folds, 5)))
    """
    --------------------------------------------- pre-training model---------------------------------------------------
    """
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    splits_train, splits_test = [], []
    for (a, b) in kfold.split(np.arange(n_subjects)):
        splits_train.append(a)
        splits_test.append(b)

    start = time.time()
    for fold in range(folds):
        idx_train, idx_test = splits_train[fold], splits_test[fold]
        trainset = [SeqEEGDataset(data[i], labels[i], n_seqlen, tf_epoch) for i in idx_train]
        train_dataset = torch.utils.data.ConcatDataset(trainset)
        train_dataloader = MultiEpochsDataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=4, pin_memory=True)
        del trainset, train_dataset

        train_model = False
        if train_model:
            model = []
            optimizer = []
            for i in range(model_num):
                model.append(BayesianConvNet(n_timepoints, n_seqlen, n_classes, bayesian, mc_drop, device).to(device))
            parameters = sum(p.numel() for p in model[0].parameters() if p.requires_grad)
            print("Num of trainable parameters per model:", parameters)
            parameters *= model_num
            print("Num of trainable parameters:", parameters)
            print()

            for i in range(model_num):
                optimizer.append(torch.optim.AdamW(model[i].parameters(), lr, betas=(0.9, 0.95)))
            start = time.time()
            for epoch in range(n_epochs):
                for i in range(model_num):
                    adjust_learning_rate(optimizer[i], epoch, n_epochs, lr)
                train_epoch(train_dataloader, model_num, model, optimizer, device, kl_weight)
                print(f"Fold: {fold}, Epoch: {epoch}| Elapsed time = {as_minutes(time.time() - start)}")
                print()
        else:
            model = torch.load(savepath + str(fold))
            for j in range(len(model)):
                model[j] = model[j].to(device)
        del train_dataloader
        """
        -------------------------------------Uncertainty-guided active learning-----------------------------------------
        """
        lr_mul = 0.3
        for k in range(len(idx_test)):
            cur_model = torch.load(savepath + str(fold))
            for j in range(len(cur_model)):
                cur_model[j] = cur_model[j].to(device)
            testset = SeqEEGDataset(data[idx_test[k]], labels[idx_test[k]], n_seqlen, tf_epoch)
            test_dataset = torch.utils.data.ConcatDataset([testset])
            test_dataloader = MultiEpochsDataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                    num_workers=4, pin_memory=True)
            del testset, test_dataset
            """
            ----------------------------------uncertainty-guided systematic sampling-----------------------------------
            """
            experts_batch_x, experts_batch_y, remained_batch_x, remained_batch_y, weight \
                = experts_set_quantified_uncertainty(test_dataloader, model, model_num, 50, False, 0.1, device)
            del test_dataloader
            """
            --------------------------------------------fine-tuning model-----------------------------------------------
            """
            experts_batch_x = torch.split(experts_batch_x, batch_size, dim=0)
            experts_batch_y = torch.split(experts_batch_y, batch_size, dim=0)
            experts_dataloader = tuple(zip(experts_batch_x, experts_batch_y))
            del experts_batch_x, experts_batch_y
            remained_batch_x = torch.split(remained_batch_x, batch_size, dim=0)
            remained_batch_y = torch.split(remained_batch_y, batch_size, dim=0)
            remained_dataloader = tuple(zip(remained_batch_x, remained_batch_y))
            del remained_batch_x, remained_batch_y
            experts_optimizer = []
            for i in range(model_num):
                experts_optimizer.append(torch.optim.AdamW(cur_model[i].parameters(), lr*lr_mul, betas=(0.9, 0.95)))
            for epoch in range(n_epochs):
                train(experts_dataloader, 1, [cur_model[0]], [experts_optimizer[0]], device, kl_weight)
                train(experts_dataloader, 1, [cur_model[1]], [experts_optimizer[1]], device, kl_weight)
            del experts_optimizer, experts_dataloader
            """
            --------------------------------------testing fine-tuned model----------------------------------------------
            """
            # (cur_true_num, cur_test_num, cur_tp, cur_fp, cur_tn, cur_fn, cur_a, cur_b) \
            #     = test(remained_dataloader, cur_model, model_num, device)
            (cur_true_num, cur_test_num, cur_tp, cur_fp, cur_tn, cur_fn, cur_a, cur_b) \
                = test_entropy_uncertainty(remained_dataloader, cur_model, model_num, 0.8/0.9, device)
            """
            ---------------------------------------- evaluation metrics -----------------------------------------------
            """
            true_num1[fold] += cur_true_num
            test_num1[fold] += cur_test_num
            tp1[fold] += cur_tp
            fp1[fold] += cur_fp
            tn1[fold] += cur_tn
            fn1[fold] += cur_fn
            a1[fold] += cur_a
            b1[fold] += cur_b
            print('current_acc:' + str(true_num1[fold] / test_num1[fold]))
            del remained_dataloader, cur_model
        print('>>>>> fold:' + str(fold) + ' acc:' + str(true_num1[fold]/test_num1[fold]))
        print(f"Fold: {fold}| Elapsed time = {as_minutes(time.time() - start)}")
        print()
    """
    ---------------------------------------------calculating evaluation metrics-----------------------------------------
    """
    test_num1 = sum(test_num1) / folds
    true_num1 = sum(true_num1) / folds
    tp1, tn1, fp1, fn1 = torch.sum(tp1, dim=0), torch.sum(tn1, dim=0), torch.sum(fp1, dim=0), torch.sum(fn1, dim=0)
    a1, b1 = torch.sum(a1, dim=0), torch.sum(b1, dim=0)
    pr1, re1 = torch.zeros(5), torch.zeros(5)
    f11 = torch.zeros(5)
    for y in range(5):
        pr1[y] = torch.sum(tp1[y]) / (torch.sum(tp1[y]) + torch.sum(fp1[y]))
        re1[y] = torch.sum(tp1[y]) / (torch.sum(tp1[y]) + torch.sum(fn1[y]))
    for y in range(5):
        f11[y] = 2 * pr1[y] * re1[y] / (pr1[y] + re1[y])
    pe1 = torch.sum(a1 * b1) / test_num1 / test_num1 / folds / folds
    mf1 = 0.4 * (torch.sum(pr1 * re1 / (pr1 + re1))) * 100
    k1 = 1 - (1 - true_num1 / test_num1 / 100) / (1 - pe1)
    print('acc:  ' + str(true_num1 / test_num1))
    print('mf1:  ' + str(mf1.item()))
    print('k:    ' + str(k1.item()))
    print('W:    ' + str(f11[0].item() * 100))
    print('N1:   ' + str(f11[1].item() * 100))
    print('N2:   ' + str(f11[2].item() * 100))
    print('N3:   ' + str(f11[3].item() * 100))
    print('REM:  ' + str(f11[4].item() * 100))


if __name__ == '__main__':
    main()