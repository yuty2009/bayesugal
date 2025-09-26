import torch.nn.functional as F
import numpy as np
import torch
import tqdm
import sys


def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_epoch(dataloader, model, weight, optimizer, device):

    accu_list, loss_list = [], []
    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # batch_y = torch.eye(n_classes)[batch_y]

        batch_yp, loss_kl = model(batch_x, sample=True)

        # model.update_weight(batch_x, batch_y, sample=False)
        celoss = F.cross_entropy(batch_yp, batch_y.long())
        trainloss = celoss + loss_kl*weight
        trainaccu = accuracy(batch_yp, batch_y)[0]
        loss_list.append(trainloss.item() / batch_x.shape[0])
        accu_list.append(trainaccu.item())

        optimizer.zero_grad()
        trainloss.backward()
        optimizer.step()

        # info = "Train Epoch: [{}] | lr: [{:.6f}] | Loss: [{:.4f}] " \
        #        "| Acc: [{:.2f}%]".format(
        #     epoch, optimizer.param_groups[0]['lr'], np.mean(loss_list),
        #     np.mean(accu_list))

        info = "Train | Loss: [{:.4f}] | Acc: [{:.2f}%]".format(
            np.mean(loss_list), np.mean(accu_list))
        data_bar.set_description(info)
        data_bar.update()

    data_bar.close()

    return np.mean(loss_list), np.mean(accu_list)


def test_epoch(dataloader, model, device):

    accu_list, loss_list = [], []
    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # batch_y = torch.eye(n_classes)[batch_y]

        with torch.no_grad():
            batch_yp, loss_kl = model(batch_x, sample=False)
        testloss = F.cross_entropy(batch_yp, batch_y.long())
        testaccu = accuracy(batch_yp, batch_y)[0]
        loss_list.append(testloss.item() / batch_x.shape[0])
        accu_list.append(testaccu.item())

        info = "Test  | Loss: [{:.4f}] | Acc: [{:.2f}%]".format(
            np.mean(loss_list), np.mean(accu_list))
        data_bar.set_description(info)
        data_bar.update()

    data_bar.close()

    return np.mean(loss_list), np.mean(accu_list)


def test_mcmc(dataloader, model, device):

    accu_list, loss_list = [], []
    data_bar = tqdm.tqdm(total=len(dataloader), file=sys.stdout)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # batch_y = torch.eye(n_classes)[batch_y]

        with torch.no_grad():
            batch_yp, loss_kl = model.predict_mcmc(batch_x)
        testloss = F.cross_entropy(batch_yp, batch_y.long())
        testaccu = accuracy(batch_yp, batch_y)[0]
        loss_list.append(testloss.item() / batch_x.shape[0])
        accu_list.append(testaccu.item())

        info = "  >>>>  Test  | Loss: [{:.5f}] | Acc: [{:.2f}%]  <<<<  ".format(
            np.mean(loss_list), np.mean(accu_list))
        data_bar.set_description(info)
        data_bar.update()

    data_bar.close()

    return np.mean(loss_list), np.mean(accu_list)
