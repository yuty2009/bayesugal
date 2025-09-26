import torch.nn.functional as F
import math
import numpy as np
import torch
import tqdm
import sys
from torch.cuda.amp import autocast, GradScaler

device_ids = list(range(torch.cuda.device_count()))


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def accuracy(output, target, train=True, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        maxmax, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        res = res[0]
        if not train:
            tp, fp, tn, fn = torch.zeros(5), torch.zeros(5), torch.zeros(5), torch.zeros(5)
            a, b = torch.zeros(5), torch.zeros(5)
            for i in range(pred.size(1)):
                for j in range(5):
                    if int(pred[0, i]) == j:
                        if int(target[i]) == j:
                            tp[j] += 1
                        else:
                            fp[j] += 1
                    else:
                        if int(target[i]) == j:
                            fn[j] += 1
                        else:
                            tn[j] += 1
            for i in range(pred.size(1)):
                a[pred[0, i]] += 1
                b[target[i]] += 1
            return res, tp, fp, tn, fn, a, b
        else:
            return res


def output_multi_models(output, output_std, output_num, device):
    '''
    final output when model_num > 1
    '''
    if torch.sum(output_std[0]).item() != 0:
        with torch.no_grad():
            mu, std, weight = [], [], []
            final_output = torch.zeros_like(output[0]).to(device)
            for i in range(output_num):
                mu.append(output[i] / torch.sum(output[i], dim=1, keepdim=True))
                std.append(output_std[i] / torch.sum(output[i], dim=1, keepdim=True))
            for i in range(output_num):
                weight.append(torch.pow(mu[i], 0) / torch.pow(std[i], 1))
                final_output += weight[i] * mu[i]
    else:
        final_output = output[0]

    return final_output


def train_epoch(dataloader, model_num, model, optimizer, device, weight):
    '''
    for each batch, only one model is updated during training, 
    the models are updated in turn
    '''
    accu, ce_loss, kl_loss, train_num \
        = torch.zeros(model_num), torch.zeros(model_num), torch.zeros(model_num), torch.zeros(model_num)
    accu, ce_loss, kl_loss, train_num = accu.to(device), ce_loss.to(device), kl_loss.to(device), train_num.to(device)
    scaler = []
    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)
    for i in range(model_num):
        scaler.append(GradScaler())

    pi = 1
    current_model = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        current_train_num = len(batch_y)
        train_num[current_model] += current_train_num
        with autocast():
            batch_yp, klloss = model[current_model](batch_x)
            celoss = F.cross_entropy(batch_yp, batch_y.long())
            trainloss = celoss + pi * weight * klloss
            trainaccu = accuracy(batch_yp, batch_y, train=True)

        scaler[current_model].scale(trainloss).backward()
        scaler[current_model].step(optimizer[current_model])
        scaler[current_model].update()

        ce_loss[current_model] += celoss
        kl_loss[current_model] += klloss
        accu[current_model] += trainaccu * current_train_num

        data_bar.update()
        pi /= 2
        current_model += 1
        if current_model == model_num:
            current_model = 0
    data_bar.close()
    accu = accu / train_num
    ce_loss = ce_loss / train_num
    kl_loss = kl_loss / train_num
    for i in range(model_num):
        print("    [Accu" + str(i) + "]" + str(round(accu[i].item(), 2)) + "% | [ce_loss" + str(i) + "]" + str(
            round(ce_loss[i].item(), 8)) + " | [kl_loss" + str(i) + "]" + str(round(kl_loss[i].item(), 8)))

    return 0


def train(dataloader, model_num, model, optimizer, device, weight):
    '''
    during training, all models are updated for each batch
    '''
    accu, ce_loss, kl_loss, train_num \
        = torch.zeros(model_num), torch.zeros(model_num), torch.zeros(model_num), torch.zeros(model_num)
    accu, ce_loss, kl_loss, train_num = accu.to(device), ce_loss.to(device), kl_loss.to(device), train_num.to(device)
    scaler = []
    # data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)
    for i in range(model_num):
        scaler.append(GradScaler())

    pi = 1

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        for i in range(model_num):
            current_train_num = len(batch_y)
            train_num[i] += current_train_num
            with autocast():
                batch_yp, klloss = model[i](batch_x)
                celoss = F.cross_entropy(batch_yp, batch_y.long())
                trainloss = celoss + pi * weight * klloss
                trainaccu = accuracy(batch_yp, batch_y, train=True)

            scaler[i].scale(trainloss).backward()
            scaler[i].step(optimizer[i])
            scaler[i].update()

            ce_loss[i] += celoss
            kl_loss[i] += klloss
            accu[i] += trainaccu * current_train_num

        # data_bar.update()
        pi /= 2
    # data_bar.close()
    accu = accu / train_num
    ce_loss = ce_loss / train_num
    kl_loss = kl_loss / train_num
    # for i in range(model_num):
    #     print("    [Accu" + str(i) + "]" + str(round(accu[i].item(), 2)) + "% | [ce_loss" + str(i) + "]" + str(
    #         round(ce_loss[i].item(), 8)) + " | [kl_loss" + str(i) + "]" + str(round(kl_loss[i].item(), 8)))

    return 0


def experts_set_entropy(dataloader, model, model_num, n_samples, random, proportion, device):
    '''
    return the selected samples and remaining samples by stage 1 uncertainty-guided systematic sampling,
    uncertainty used here is entropy,
    n_samples: number of forward passed samples,
    random: whether randomly select samples,
    proportion: the proportion of selected samples,
    '''
    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)
    experts_batch_x_list, experts_batch_y_list, remained_batch_x_list, remained_batch_y_list = [], [], [], []
    weight_list = []
    for k, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output, output_std = [], []

        # Use torch.no_grad to avoid calculating gradients
        with torch.no_grad():
            for i in range(model_num):
                batch_yp, _ = model[i].influence(batch_x, n_samples=int(n_samples / model_num))
                entropy = -torch.sum(batch_yp * torch.log(batch_yp + 1e-10), dim=1)
                output.append(batch_yp.half())
                output_std.append(entropy.half())

            weight = torch.zeros(len(batch_y), device=device)

            for i in range(model_num):
                weight += output_std[i]

            sorted_indices = torch.argsort(weight, descending=False)
            random_indices = torch.arange(0, len(sorted_indices))
            if random:
                selected_indices = random_indices[::int(1 / proportion)]
            else:
                selected_indices = sorted_indices[::int(1 / proportion)]

            remain_indices = list(set(sorted_indices.tolist()) - set(selected_indices.tolist()))
            remain_indices = sorted(remain_indices, key=lambda x: sorted_indices.tolist().index(x))

            experts_batch_x_list.append(batch_x[selected_indices])
            experts_batch_y_list.append(batch_y[selected_indices])
            remained_batch_x_list.append(batch_x[remain_indices])
            remained_batch_y_list.append(batch_y[remain_indices])
            del batch_x, batch_y

        del output, output_std, weight
        data_bar.update()
    experts_batch_x_list = torch.concat(experts_batch_x_list)
    experts_batch_y_list = torch.concat(experts_batch_y_list)
    remained_batch_x_list = torch.concat(remained_batch_x_list)
    remained_batch_y_list = torch.concat(remained_batch_y_list)
    data_bar.close()

    return experts_batch_x_list, experts_batch_y_list, remained_batch_x_list, remained_batch_y_list, weight_list


def experts_set_quantified_uncertainty(dataloader, model, model_num, n_samples, random, proportion, device):
    '''
    return the selected samples and remaining samples by stage 1 uncertainty-guided systematic sampling,
    uncertainty used here is the proposed quantified uncertainty,
    n_samples: number of forward passed samples,
    random: whether randomly select samples,
    proportion: the proportion of selected samples,
    '''
    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)
    experts_batch_x_list, experts_batch_y_list, remained_batch_x_list, remained_batch_y_list = [], [], [], []
    weight_list = []
    for k, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        output, output_std = [], []

        # Use torch.no_grad to avoid calculating gradients
        with torch.no_grad():
            for i in range(model_num):
                batch_yp, std = model[i].influence(batch_x, n_samples=int(n_samples / model_num))
                bayes_uncert = torch.pow(std/batch_yp, 1)
                quantified_uncertainty = -torch.sum(batch_yp * torch.log(batch_yp + 1e-10) * bayes_uncert, dim=1)
                output.append(batch_yp.half())
                output_std.append(quantified_uncertainty.half())

            weight = torch.zeros(len(batch_y), device=device)

            for i in range(model_num):
                weight += output_std[i]

            sorted_indices = torch.argsort(weight, descending=False)
            random_indices = torch.arange(0, len(sorted_indices))
            if random:
                selected_indices = random_indices[::int(1 / proportion)]
            else:
                selected_indices = sorted_indices[::int(1 / proportion)]

            remain_indices = list(set(sorted_indices.tolist()) - set(selected_indices.tolist()))
            remain_indices = sorted(remain_indices, key=lambda x: sorted_indices.tolist().index(x))

            experts_batch_x_list.append(batch_x[selected_indices])
            experts_batch_y_list.append(batch_y[selected_indices])
            remained_batch_x_list.append(batch_x[remain_indices])
            remained_batch_y_list.append(batch_y[remain_indices])
            del batch_x, batch_y

        del output, output_std, weight
        data_bar.update()
    experts_batch_x_list = torch.concat(experts_batch_x_list)
    experts_batch_y_list = torch.concat(experts_batch_y_list)
    remained_batch_x_list = torch.concat(remained_batch_x_list)
    remained_batch_y_list = torch.concat(remained_batch_y_list)
    data_bar.close()

    return experts_batch_x_list, experts_batch_y_list, remained_batch_x_list, remained_batch_y_list, weight_list


def experts_set_entropy_uncertainty_index(dataloader, model, model_num, n_samples, random, proportion, device):
    '''
    return the indices of the selected samples and remaining samples,
    uncertainty used here is the proposed Quantified Uncertainty,
    n_samples: number of forward passed samples,
    random: whether randomly select samples,
    proportion: the proportion of selected samples in Stage 1,
    '''
    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)
    pretrained_network_output, clinician1_i, remained_network_output, remained_i = [], [], [], []

    batch_num = 0
    for k, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_i = torch.arange(batch_num, batch_num+len(batch_y)).to(device)
        batch_num = batch_num+len(batch_y)
        output, output_std = [], []

        with torch.no_grad():
            for i in range(model_num):
                batch_yp, std = model[i].influence(batch_x, n_samples=int(n_samples / model_num))
                bayes_uncert = torch.pow(std/batch_yp, 1)
                entropy = -torch.sum(batch_yp * torch.log(batch_yp + 1e-10) * bayes_uncert, dim=1)
                output.append(batch_yp.half())
                output_std.append(entropy.half())
            weight = torch.zeros(len(batch_y), device=device)

        y = torch.zeros_like(output[0])
        for i in range(model_num):
            y += output[i] / output_std[i].unsqueeze(-1).repeat(1, 5)

        for i in range(model_num):
            weight += output_std[i]

        sorted_indices = torch.argsort(weight, descending=False)

        random_indices = torch.arange(0, len(sorted_indices))
        if random:
            selected_indices = random_indices[::int(1 / proportion)]
        else:
            selected_indices = sorted_indices[::int(1 / proportion)]

        remain_indices = list(set(sorted_indices.tolist()) - set(selected_indices.tolist()))
        remain_indices = sorted(remain_indices, key=lambda x: sorted_indices.tolist().index(x))

        pretrained_network_output.append(y)
        clinician1_i.append(batch_i[selected_indices])
        remained_i.append(batch_i[remain_indices])
        del batch_x, batch_y, batch_i

        del output, output_std, weight
        data_bar.update()
    pretrained_network_output = torch.concat(pretrained_network_output)
    clinician1_i = torch.concat(clinician1_i)
    remained_i = torch.concat(remained_i)
    data_bar.close()

    return pretrained_network_output, clinician1_i, remained_i


def test(dataloader, model, model_num, device):
    loss = torch.zeros(1).to(device)
    true_num_per_model = torch.zeros(model_num).to(device)
    true_num, test_num = 0, 0
    tp, fp, tn, fn, a, b = 0, 0, 0, 0, 0, 0

    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        current_test_num = batch_x.shape[0]
        output, output_std = [], []
        for i in range(model_num):
            with torch.no_grad():
                batch_yp, std = model[i].influence(batch_x, n_samples=int(200 / model_num))
                current_test_accu = accuracy(batch_yp, batch_y, train=False)[0]
                true_num_per_model[i] += current_test_accu * current_test_num
                output.append(batch_yp)
                output_std.append(std)
        batch_yp = output_multi_models(output, output_std, model_num, device)
        testloss = F.cross_entropy(batch_yp, batch_y.long())
        testaccu, tp1, fp1, tn1, fn1, a1, b1 = accuracy(batch_yp, batch_y, train=False)

        loss += testloss * current_test_num
        true_num += testaccu * current_test_num
        test_num += current_test_num
        tp += tp1
        fp += fp1
        tn += tn1
        fn += fn1
        a += a1
        b += b1
        info = ">>>> Test | Loss: [{:.6f}] | Acc: [{:.2f}%] <<<<".format(
            loss.item() / test_num, true_num / test_num)
        data_bar.set_description(info)
        data_bar.update()
    data_bar.close()
    for i in range(model_num):
        print("[Accu" + str(i) + "] " + str(round((true_num_per_model[i] / test_num).item(), 2)) + "%")
    return true_num, test_num, tp, fp, tn, fn, a, b


def test_entropy_uncertainty(dataloader, model, model_num, proportion, device):
    '''
    test the remained samples 
    uncertainty used here is entropy,
    proportion: the proportion of selected samples in Stage 2,
    '''
    true_num_per_model = torch.zeros(model_num).to(device)
    true_num, test_num = 0, 0
    tp, fp, tn, fn, a, b = 0, 0, 0, 0, 0, 0

    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        current_test_num = batch_x.shape[0]
        output, output_uncertainty = [], []
        for i in range(model_num):
            with torch.no_grad():
                batch_yp, _ = model[i].influence(batch_x, n_samples=int(200 / model_num))
                batch_yp = F.softmax(batch_yp, dim=1)
                entropy = -torch.sum(batch_yp * torch.log(batch_yp + 1e-10), dim=1)
                current_test_accu = accuracy(batch_yp, batch_y, train=False)[0]
                true_num_per_model[i] += current_test_accu * current_test_num
                output.append(batch_yp)
                output_uncertainty.append(entropy)
        true_num1, test_num1, tp1, fp1, tn1, fn1, a1, b1 = eval_metrics_uncertainty(output, batch_y, output_uncertainty, model_num,
                                                                               proportion, device)

        true_num += true_num1
        test_num += test_num1
        tp += tp1
        fp += fp1
        tn += tn1
        fn += fn1
        a += a1
        b += b1
        info = ">>>> Test | Acc: [{:.2f}%] <<<<".format(true_num / test_num * 100)
        data_bar.set_description(info)
        data_bar.update()
    data_bar.close()
    for i in range(model_num):
        print("[Accu" + str(i) + "] " + str(round((true_num_per_model[i] / test_num * proportion).item(), 2)) + "%")
    return true_num*100, test_num, tp, fp, tn, fn, a, b


def test_quantified_uncertainty(dataloader, model, model_num, proportion, device):
    '''
    test the remained samples 
    uncertainty used here is the proposed quantified uncertainty,
    proportion: the proportion of selected samples in Stage 2,
    '''
    true_num_per_model = torch.zeros(model_num).to(device)
    true_num, test_num = 0, 0
    tp, fp, tn, fn, a, b = 0, 0, 0, 0, 0, 0

    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        current_test_num = batch_x.shape[0]
        output, output_uncertainty = [], []
        for i in range(model_num):
            with torch.no_grad():
                batch_yp, std = model[i].influence(batch_x, n_samples=int(200 / model_num))
                batch_yp = F.softmax(batch_yp, dim=1)
                bayes_uncert = torch.pow(std / batch_yp, 0.7)
                quantified_uncertainty = -torch.sum(batch_yp * torch.log(batch_yp + 1e-10) * bayes_uncert, dim=1)
                current_test_accu = accuracy(batch_yp, batch_y, train=False)[0]
                true_num_per_model[i] += current_test_accu * current_test_num
                output.append(batch_yp)
                output_uncertainty.append(quantified_uncertainty)
        true_num1, test_num1, tp1, fp1, tn1, fn1, a1, b1 = eval_metrics_uncertainty(output, batch_y, output_uncertainty, model_num,
                                                                               proportion, device)

        true_num += true_num1
        test_num += test_num1
        tp += tp1
        fp += fp1
        tn += tn1
        fn += fn1
        a += a1
        b += b1
        info = ">>>> Test | Acc: [{:.2f}%] <<<<".format(true_num / test_num * 100)
        data_bar.set_description(info)
        data_bar.update()
    data_bar.close()
    for i in range(model_num):
        print("[Accu" + str(i) + "] " + str(round((true_num_per_model[i] / test_num * proportion).item(), 2)) + "%")
    return true_num*100, test_num, tp, fp, tn, fn, a, b


def test_quantified_uncertainty_index(dataloader, index, model, model_num, proportion, device): 
    '''
    uncertainty-guided systematic sampling,
    proportion: the proportion of selected samples in Stage 2,
    '''
    index_acp, index_rej = [], []
    net_output, cli2_gt = [], []

    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)
    batch_num = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_num += len(batch_y)
        output, output_std = [], []
        for i in range(model_num):
            with torch.no_grad():
                batch_yp, std = model[i].influence(batch_x, n_samples=int(200 / model_num))
                batch_yp = F.softmax(batch_yp, dim=1)
                bayes_uncert = torch.pow(std / batch_yp, 0.7)
                entropy = -torch.sum(batch_yp * torch.log(batch_yp + 1e-10) * bayes_uncert, dim=1)

                output.append(batch_yp)
                output_std.append(entropy)
        network_output, cli2_ground_truth, accepted_index, rejected_index\
            = uncertainty_index(output, batch_y, output_std, model_num, proportion, index)

        net_output.append(network_output)
        cli2_gt.append(cli2_ground_truth)
        index_acp.append(accepted_index)
        index_rej.append(rejected_index)

        data_bar.update()
    data_bar.close()

    net_output = torch.concat(net_output)
    cli2_gt = torch.concat(cli2_gt)
    index_acp = torch.concat(index_acp)
    index_rej = torch.concat(index_rej)
    return net_output, cli2_gt, index_acp, index_rej


def eval_metrics_uncertainty(output, target, output_uncertainty, output_num, proportion, device):
    '''
    return the evaluation metrics of the remained samples after stage 2 sampling,
    proportion: the proportion of selected samples in Stage 2,
    '''
    with torch.no_grad():

        y = torch.zeros_like(output[0])
        for i in range(output_num):
            y += output[i] / output_uncertainty[i].unsqueeze(-1).repeat(1, 5)
        weight = torch.zeros_like(output_uncertainty[0])
        for i in range(output_num):
            weight += output_uncertainty[i]
        sorted_indices = torch.argsort(weight, descending=False)
        sorted_output = y[sorted_indices]
        sorted_target = target[sorted_indices]
        accepted_output = sorted_output[:int(len(target) * proportion)]
        accepted_target = sorted_target[:int(len(target) * proportion)]
        if len(accepted_target) == 0:
            true_num = torch.zeros([])
            tp, fp, tn, fn = torch.zeros(5), torch.zeros(5), torch.zeros(5), torch.zeros(5)
            a, b = torch.zeros(5), torch.zeros(5)
            test_num = torch.zeros([])
            test_num = test_num.long()
        else:
            true_num, tp, fp, tn, fn, a, b = accuracy(accepted_output, accepted_target, train=False)
            true_num = true_num * int(len(target) * proportion) / 100
            test_num = torch.tensor(int(len(target) * proportion)).to(device)

    return true_num, test_num, tp, fp, tn, fn, a, b


def uncertainty_index(output, target, output_std, output_num, proportion, index):
    '''
    return the indices of selected and remained samples after stage 2 sampling,
    proportion: the proportion of selected samples in Stage 2,
    '''
    with torch.no_grad():

        y = torch.zeros_like(output[0])
        for i in range(output_num):
            y += output[i] / output_std[i].unsqueeze(-1).repeat(1, 5)
        weight = torch.zeros_like(output_std[0])
        for i in range(output_num):
            weight += output_std[i]
        sorted_indices = torch.argsort(weight, descending=False)
        sorted_index = index[sorted_indices]
        sorted_output = y[sorted_indices]
        sorted_target = target[sorted_indices]
        accepted_output = sorted_output[:int(len(target) * proportion)]
        rejected_gt = sorted_target[int(len(target) * proportion):]
        accepted_index = sorted_index[:int(len(target) * proportion)]
        rejected_index = sorted_index[int(len(target) * proportion):]

    return accepted_output, rejected_gt, accepted_index, rejected_index


def adjust_learning_rate(optimizer, epoch, epochs, lr):
    warmup_epochs = epochs // 10
    min_lr = 0
    if epoch < warmup_epochs:
        lr = lr * (epoch + 1) / warmup_epochs
    else:
        lr = lr
        lr = min_lr + (lr - min_lr) * 0.5 * \
             (1. + math.cos(math.pi * ((epoch + 1) - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

