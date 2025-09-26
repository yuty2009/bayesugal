import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tools import *
from torch.utils.data import random_split
import torch.utils.data.distributed
from pre.utils import *
from EDFreader.sleepreader import *
from model import BayesianConvNet
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from pathlib import Path

def main():
    device_id = 0
    n_seqlen = 20
    n_classes = 5
    n_epochs = 200
    lr = 12e-4
    batch_size = 1024
    kl_weight = 2e-8
    """
    >>>> Chose Model Type
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
    >>>> Chose dataset: 0 - edf20, 1 - edf78, 2- MASS, 3- physio2018
    """
    dataset = 0
    dataset_paths = {
        0: {'path': 'sleep-edf-20', 'n_subjects': 39, 'folds': 10},
        1: {'path': 'sleep-edf-78', 'n_subjects': 153, 'folds': 10},
        2: {'path': 'MASS', 'n_subjects': 200, 'folds': 10},
        3: {'path': 'physionet2018', 'n_subjects': 994, 'folds': 5}
    }
    """
    --------------------------------------------train_total---------------------------------------
    """
    data_saved = True
    if not data_saved:
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
        kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
        splits_train, splits_test = [], []
        for (a1, b1) in kfold.split(np.arange(n_subjects)):
            splits_train.append(a1)
            splits_test.append(b1)

        start = time.time()
    
        fold = 0
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
        ----------------------------------------------------------------------------------------------
        """
        lr_mul = 0.3
        k = 0
        cur_model = torch.load(savepath + str(fold))
        for j in range(len(cur_model)):
            cur_model[j] = cur_model[j].to(device)
        testset = SeqEEGDataset(data[idx_test[k]], labels[idx_test[k]], n_seqlen, tf_epoch)
        test_dataset = torch.utils.data.ConcatDataset([testset])
        test_dataloader_index = MultiEpochsDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        del testset, test_dataset
        """
        -------------------------------------------------------------------------------------------
        """
        pretrained_network_output, cli1_index, rmn1_index \
            = experts_set_entropy_uncertainty_index(test_dataloader_index, model, model_num,  50, False, 0.05, device)
        all_data_x = []
        ground_truth_y = []
        all_data_index = []
        cur_index = 0
        for data, target in test_dataloader_index:
            all_data_x.append(data)
            ground_truth_y.append(target)
            all_data_index.append(torch.arange(cur_index, cur_index+len(target)))
            cur_index += len(target)
        all_data_x = torch.concat(all_data_x, dim=0).to(device)
        ground_truth_y = torch.concat(ground_truth_y, dim=0).to(device)
        all_data_index = torch.concat(all_data_index)

        cli1_data = all_data_x[cli1_index]
        cli1_ground_truth = ground_truth_y[cli1_index]
        clinician1_data1 = torch.split(cli1_data, batch_size, dim=0)
        clinician1_ground_truth1 = torch.split(cli1_ground_truth, batch_size, dim=0)
        clinician1_dataloader = tuple(zip(clinician1_data1, clinician1_ground_truth1))
        del cli1_data

        rmn1_data = all_data_x[rmn1_index]
        rmn1_ground_truth = ground_truth_y[rmn1_index]
        remained1_data1 = torch.split(rmn1_data, batch_size, dim=0)
        remained1_ground_truth1 = torch.split(rmn1_ground_truth, batch_size, dim=0)
        remained1_dataloader = tuple(zip(remained1_data1, remained1_ground_truth1))
        del rmn1_data

        clinician1_optimizer = []
        for i in range(model_num):
            clinician1_optimizer.append(torch.optim.AdamW(cur_model[i].parameters(), lr*lr_mul, betas=(0.9, 0.95)))
        for epoch in range(n_epochs):
            train(clinician1_dataloader, 1, [cur_model[0]], [clinician1_optimizer[0]], device, kl_weight)
            train(clinician1_dataloader, 1, [cur_model[1]], [clinician1_optimizer[1]], device, kl_weight)
        del clinician1_optimizer, clinician1_dataloader
        """
        --------------------------------------------------------------------------------------------
        """
        (fine_tuned_net_output, cli2_ground_truth, rmn2_index, cli2_index) \
            = test_quantified_uncertainty_index(remained1_dataloader, rmn1_index, cur_model, model_num, 0.9/0.95, device)
        del remained1_dataloader, cur_model

        pretrained_network_output = torch.argmax(pretrained_network_output, dim=1)

        # 获取预测结果与真实标签不同的位置索引
        different_indices = torch.where(pretrained_network_output != ground_truth_y)[0]

        fine_tuned_index = torch.concat([rmn2_index, cli2_index, cli1_index], dim=0)
        sorted_indices = torch.argsort(fine_tuned_index, descending=False)
        fine_tuned_index = fine_tuned_index[sorted_indices]

        fine_tuned_net_output = torch.argmax(fine_tuned_net_output, dim=1)
        fine_tuned_net_output = torch.concat([fine_tuned_net_output, cli2_ground_truth, cli1_ground_truth], dim=0)
        fine_tuned_net_output = fine_tuned_net_output[sorted_indices]
        
        # 计算微调后的预测准确性
        fine_tuned_different_indices = torch.where(fine_tuned_net_output != ground_truth_y)[0]
        torch.save(cli1_index, 'F:\\bayessleepnet\\figures\\data\\cli1_index')
        torch.save(cli1_ground_truth, 'F:\\bayessleepnet\\figures\\data\\cli1_ground_truth')
        torch.save(cli2_index, 'F:\\bayessleepnet\\figures\\data\\cli2_index')
        torch.save(cli2_ground_truth, 'F:\\bayessleepnet\\figures\\data\\cli2_ground_truth')
        torch.save(all_data_index, 'F:\\bayessleepnet\\figures\\data\\all_data_index')
        torch.save(ground_truth_y, 'F:\\bayessleepnet\\figures\\data\\ground_truth_y')
        torch.save(different_indices, 'F:\\bayessleepnet\\figures\\data\\different_indices')
        torch.save(fine_tuned_different_indices, 'F:\\bayessleepnet\\figures\\data\\fine_tuned_different_indices')
        torch.save(pretrained_network_output, 'F:\\bayessleepnet\\figures\\data\\pretrained_network_output')
        torch.save(fine_tuned_net_output, 'F:\\bayessleepnet\\figures\\data\\fine_tuned_net_output')
    else:
        cli1_index = torch.load('F:\\bayessleepnet\\figures\\data\\cli1_index')
        cli1_ground_truth = torch.load('F:\\bayessleepnet\\figures\\data\\cli1_ground_truth')
        cli2_index = torch.load('F:\\bayessleepnet\\figures\\data\\cli2_index')
        cli2_ground_truth = torch.load('F:\\bayessleepnet\\figures\\data\\cli2_ground_truth')
        all_data_index = torch.load('F:\\bayessleepnet\\figures\\data\\all_data_index')
        ground_truth_y = torch.load('F:\\bayessleepnet\\figures\\data\\ground_truth_y')
        different_indices = torch.load('F:\\bayessleepnet\\figures\\data\\different_indices')
        fine_tuned_different_indices = torch.load('F:\\bayessleepnet\\figures\\data\\fine_tuned_different_indices')
        pretrained_network_output = torch.load('F:\\bayessleepnet\\figures\\data\\pretrained_network_output')
        fine_tuned_net_output = torch.load('F:\\bayessleepnet\\figures\\data\\fine_tuned_net_output')
    
    # 计算错误点数量和总体准确率
    pretrained_error_count = len(different_indices)
    finetuned_error_count = len(fine_tuned_different_indices)
    total_data_count = len(all_data_index)
    
    pretrained_accuracy = (total_data_count - pretrained_error_count) / total_data_count * 100
    finetuned_accuracy = (total_data_count - finetuned_error_count) / total_data_count * 100
    
    # 计算详细的评估指标
    # 转换为numpy数组用于sklearn计算
    gt_np = ground_truth_y.cpu().numpy()
    pretrained_pred_np = pretrained_network_output.cpu().numpy()
    finetuned_pred_np = fine_tuned_net_output.cpu().numpy()
    
    # Pretrained Network 指标
    pretrained_overall_acc = accuracy_score(gt_np, pretrained_pred_np) * 100
    pretrained_mf1 = f1_score(gt_np, pretrained_pred_np, average='macro') * 100
    pretrained_kappa = cohen_kappa_score(gt_np, pretrained_pred_np) * 100
    
    # Fine-tuned Network 指标
    finetuned_overall_acc = accuracy_score(gt_np, finetuned_pred_np) * 100
    finetuned_mf1 = f1_score(gt_np, finetuned_pred_np, average='macro') * 100
    finetuned_kappa = cohen_kappa_score(gt_np, finetuned_pred_np) * 100
    
    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Total data points: {total_data_count}")
    print()
    
    print("Pretrained Network:")
    print(f"  Error points: {pretrained_error_count}")
    print(f"  Overall Accuracy: {pretrained_overall_acc:.2f}%")
    print(f"  Macro F1 (MF1): {pretrained_mf1:.2f}%")
    print(f"  Cohen's Kappa: {pretrained_kappa:.2f}%")
    print()
    
    print("Fine-tuned Network:")
    print(f"  Error points: {finetuned_error_count}")
    print(f"  Overall Accuracy: {finetuned_overall_acc:.2f}%")
    print(f"  Macro F1 (MF1): {finetuned_mf1:.2f}%")
    print(f"  Cohen's Kappa: {finetuned_kappa:.2f}%")
    print()
    
    print("Improvements (Fine-tuned vs Pretrained):")
    print(f"  Accuracy improvement: {finetuned_overall_acc - pretrained_overall_acc:.2f}%")
    print(f"  MF1 improvement: {finetuned_mf1 - pretrained_mf1:.2f}%")
    print(f"  Kappa improvement: {finetuned_kappa - pretrained_kappa:.2f}%")
    print("=" * 60)
    print()

    plt.style.use('seaborn-v0_8')
    mark_points = [
        {'x': cli1_index.cpu(),
            'y': cli1_ground_truth.cpu(),
            'color': [246 / 255, 167 / 255, 76 / 255],
            'marker': 's',
            'size': 3,
            },

        {'x': cli2_index.cpu(),
            'y': cli2_ground_truth.cpu(),
            'color': [104 / 255, 170 / 255, 73 / 255],
            'marker': 's',
            'size': 3,
            },
    ]
    # 预测错误位置的标记点
    pretrained_network_error_points = {
        'x': different_indices.cpu(),
        'y': pretrained_network_output[different_indices].cpu(),
        'color': [1.0, 0.2, 0.2],  # 红色
        'marker': 'x',
        'size': 2,  # 增加叉叉的大小以便更清楚地看到
    }
    finetuned_network_error_points = {
        'x': fine_tuned_different_indices.cpu(),
        'y': fine_tuned_net_output[fine_tuned_different_indices].cpu(),
        'color': [1.0, 0.2, 0.2],  # 红色
        'marker': 'x',
        'size': 2,  # 增加叉叉的大小以便更清楚地看到
    }
    """
    --------------------------------------------------------------------------------------------
    """
    # 创建4个垂直排列的子图
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 4), dpi=300)
    
    # 第一个图：Ground Truth
    ax1.plot(all_data_index.cpu().numpy(), ground_truth_y.cpu().numpy(),
            color=[46 / 255, 117 / 255, 182 / 255],
            linestyle='-',
            linewidth=0.6,
            alpha=1)

    # 坐标轴范围
    ax1.set_xlim(-50, 1080)
    ax1.set_ylim(-0.5, 4.5)

    # 刻度标签设置
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'],
                        fontsize=6,
                        fontfamily='sans serif',
                        horizontalalignment='right')
    # 刻度
    ax1.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=False, labelsize=6)
    ax1.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False, labeltop=False, labelsize=6, pad=2)
    ax1.tick_params(axis='both', which='both', length=3, width=0.5, direction='in')


    # 设置背景色和边框
    ax1.set_facecolor('white')
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax1.transAxes,
                                fill=False, edgecolor='black', linewidth=1)
    ax1.add_patch(rect)
    ax1.text(-0.05, 0.5, 'Ground-Truth', transform=ax1.transAxes, fontsize=7, fontweight='bold',
            rotation=90, verticalalignment='center', horizontalalignment='right')
    
    """
    --------------------------------------------------------------------------------------------
    """
    # 第二个图：Pretrained Network Output
    ax2.plot(all_data_index.cpu().numpy(), pretrained_network_output.cpu().numpy(),
                    color=[46 / 255, 117 / 255, 182 / 255],
                    linestyle='-',
                    linewidth=0.6,
                    alpha=1,
                    zorder=1)  # 设置较低的层级
    
    # 标记预测错误的位置 - 放在线条之上
    ax2.scatter(pretrained_network_error_points['x'].numpy(), pretrained_network_error_points['y'].numpy(),
                color=pretrained_network_error_points['color'],
                marker=pretrained_network_error_points['marker'],
                s=pretrained_network_error_points['size'],
                alpha=1,
                linewidths=0.5,
                zorder=2)  # 设置较高的层级，确保显示在线条之上

    # 坐标轴范围
    ax2.set_xlim(-50, 1080)
    ax2.set_ylim(-0.5, 4.5)

    # 刻度标签设置
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'],
                        fontsize=6,
                        fontfamily='sans serif',
                        horizontalalignment='right')
    # 刻度
    ax2.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=False, labelsize=6)
    ax2.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False, labeltop=False, labelsize=6, pad=2)
    ax2.tick_params(axis='both', which='both', length=3, width=0.5, direction='in')

    # 设置背景色和边框
    ax2.set_facecolor('white')
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax2.transAxes,
                                fill=False, edgecolor='black', linewidth=1)
    ax2.add_patch(rect)
    ax2.text(-0.05, 0.5, 'Pretrained', transform=ax2.transAxes, fontsize=7, fontweight='bold',
            rotation=90, verticalalignment='center', horizontalalignment='right')
    # 添加准确率标注
    ax2.text(0.02, 0.95, f'Accuracy: {pretrained_overall_acc:.2f}%\nMF1: {pretrained_mf1:.2f}%\nKappa: {pretrained_kappa/100:.3f}', 
             transform=ax2.transAxes, fontsize=7, fontweight='bold', verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    """
    --------------------------------------------------------------------------------------------
    """
    # 第三个图：Fine-tuned Network Output with Error Points
    ax3.plot(all_data_index.cpu().numpy(), fine_tuned_net_output.cpu().numpy(),
                    color=[46 / 255, 117 / 255, 182 / 255],
                    linestyle='-',
                    linewidth=0.6,
                    alpha=1,
                    zorder=1)  # 设置较低的层级

    # 绘制预测错误点 - 放在线条之上
    ax3.scatter(finetuned_network_error_points['x'].numpy(), finetuned_network_error_points['y'].numpy(),
                color=finetuned_network_error_points['color'],
                marker=finetuned_network_error_points['marker'],
                s=finetuned_network_error_points['size'],
                alpha=1,
                linewidths=0.5,
                zorder=2)  # 设置较高的层级，确保显示在线条之上

    # 坐标轴范围
    ax3.set_xlim(-50, 1080)
    ax3.set_ylim(-0.5, 4.5)

    # 刻度标签设置
    ax3.set_yticks([0, 1, 2, 3, 4])
    ax3.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'],
                        fontsize=6,
                        fontfamily='sans serif',
                        horizontalalignment='right')
    # 刻度
    ax3.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=False, labelsize=6)
    ax3.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=False, labeltop=False, labelsize=6, pad=2)
    ax3.tick_params(axis='both', which='both', length=3, width=0.5, direction='in')

    # 设置背景色和边框
    ax3.set_facecolor('white')
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax3.transAxes,
                                fill=False, edgecolor='black', linewidth=1)
    ax3.add_patch(rect)
    ax3.text(-0.05, 0.5, 'Fine-tuned', transform=ax3.transAxes, fontsize=7, fontweight='bold',
            rotation=90, verticalalignment='center', horizontalalignment='right')
    # 添加准确率标注
    ax3.text(0.02, 0.95, f'Accuracy: {finetuned_overall_acc:.2f}%\nMF1: {finetuned_mf1:.2f}%\nKappa: {finetuned_kappa/100:.3f}', 
             transform=ax3.transAxes, fontsize=7, fontweight='bold', verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    """
    --------------------------------------------------------------------------------------------
    """
    # 第四个图：Fine-tuned Network Output with Active Learning Points
    ax4.plot(all_data_index.cpu().numpy(), fine_tuned_net_output.cpu().numpy(),
                    color=[46 / 255, 117 / 255, 182 / 255],
                    linestyle='-',
                    linewidth=0.6,
                    alpha=1)

    # 绘制主动学习标记点 - 改为垂直条形填充
    for i, point in enumerate(mark_points):
        x_coords = point['x'].numpy()
        y_coords = point['y'].numpy()
        color = point['color']
        
        for j in range(len(x_coords)):
            x_start = x_coords[j]
            x_end = x_coords[j] + 1
            y_start = -0.5
            y_end = 4.5
            
            # 创建矩形填充
            rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, facecolor=color, alpha=0.9, edgecolor='none')
            ax4.add_patch(rect)

    # 坐标轴范围
    ax4.set_xlim(-50, 1080)
    ax4.set_ylim(-0.5, 4.5)

    # 刻度标签设置
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'],
                        fontsize=6,
                        fontfamily='sans serif',
                        horizontalalignment='right')
    # 刻度
    ax4.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=False, labelsize=6)
    ax4.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=False, labelsize=6, pad=2)
    ax4.tick_params(axis='both', which='both', length=3, width=0.5, direction='in')
    # 添加x轴标签
    ax4.set_xlabel('Epoch', fontsize=7, fontweight='bold')
    # 设置背景色和边框
    ax4.set_facecolor('white')
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax4.transAxes,
                                fill=False, edgecolor='black', linewidth=1)
    ax4.add_patch(rect)
    ax4.text(-0.05, 0.5, 'Annotated', transform=ax4.transAxes, fontsize=7, fontweight='bold',
            rotation=90, verticalalignment='center', horizontalalignment='right')

    plt.tight_layout(h_pad=0.7)  # 减少子图之间的垂直间距
    plt.savefig('F:\\bayessleepnet\\figures\\sleep_stage.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    print()
    """
    -----------------------------------------------------------------------------------------------
    """


if __name__ == '__main__':
    main()
