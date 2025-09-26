import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss_calculate(p_mu, p_sigma, q_mu, q_sigma):
    q_mu = q_mu.to(p_mu.device)
    q_sigma = q_sigma.to(p_mu.device)
    loss = (torch.log(q_sigma / p_sigma) - 0.5 +
            0.5 * (p_sigma / q_sigma).pow(2) + 0.5 * ((p_mu - q_mu) / q_sigma).pow(2)).sum()
    return loss


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, device,
                 bias=True,
                 prior_w_mu=0, prior_b_mu=0, prior_w_sigma=-3, prior_b_sigma=-3):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(prior_w_mu, 0.1))
        self.prior_w_mu = prior_w_mu * torch.ones(out_features, in_features).to(device)
        self.w_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(prior_w_sigma, 0.1))
        self.prior_w_rho = prior_w_sigma * torch.ones(out_features, in_features).to(device)

        if self.bias:
            self.b_mu = nn.Parameter(torch.Tensor(out_features).normal_(prior_b_mu, 0.1))
            self.prior_b_mu = prior_b_mu * torch.ones(out_features).to(device)
            self.b_rho = nn.Parameter(torch.Tensor(out_features).normal_(prior_b_sigma, 0.1))
            self.prior_b_rho = prior_b_sigma * torch.ones(out_features).to(device)
        else:
            self.register_parameter('b_mu', None)
            self.register_parameter('b_rho', None)

    def forward(self, x, sample):
        if not sample:
            w, b = self.w_mu, self.b_mu
            loss_kl = 0
        else:
            w, b = sample_weights(self.w_mu, self.w_rho, self.b_mu, self.b_rho, self.bias)
            prior_w_sigma = 1e-6 + F.softplus(self.prior_w_rho)
            w_sigma = 1e-6 + F.softplus(self.w_rho)
            loss_kl = kl_loss_calculate(self.w_mu, w_sigma, self.prior_w_mu, prior_w_sigma)
            if self.bias:
                prior_b_sigma = 1e-6 + F.softplus(self.prior_b_rho)
                b_sigma = 1e-6 + F.softplus(self.b_rho)
                loss_kl += kl_loss_calculate(self.b_mu, b_sigma, self.prior_b_mu, prior_b_sigma)
        return F.linear(x, w, b), loss_kl


class BayesConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, device,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 prior_w_mu=0, prior_b_mu=0, prior_w_sigma=-3, prior_b_sigma=-3
                 ):
        super(BayesConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.w_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size).normal_(prior_w_mu, 0.1))
        self.prior_w_mu = prior_w_mu * torch.ones(out_channels, in_channels, kernel_size).to(device)
        self.w_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size).normal_(prior_w_sigma, 0.1))
        self.prior_w_rho = prior_w_sigma * torch.ones(out_channels, in_channels, kernel_size).to(device)

        if self.bias:
            self.b_mu = nn.Parameter(torch.Tensor(out_channels).normal_(prior_b_mu, 0.1))
            self.prior_b_mu = prior_b_mu * torch.ones(out_channels).to(device)
            self.b_rho = nn.Parameter(torch.Tensor(out_channels).normal_(prior_b_sigma, 0.1))
            self.prior_b_rho = prior_b_sigma * torch.ones(out_channels).to(device)
        else:
            self.register_parameter('b_mu', None)
            self.register_parameter('b_rho', None)

    def forward(self, x, sample):
        if not sample:
            w, b = self.w_mu, self.b_mu
            loss_kl = 0
        else:
            w, b = sample_weights(self.w_mu, self.w_rho, self.b_mu, self.b_rho, self.bias)
            prior_w_sigma = 1e-6 + F.softplus(self.prior_w_rho)
            w_sigma = 1e-6 + F.softplus(self.w_rho)
            loss_kl = kl_loss_calculate(self.w_mu, w_sigma, self.prior_w_mu, prior_w_sigma)
            if self.bias:
                prior_b_sigma = 1e-6 + F.softplus(self.prior_b_rho)
                b_sigma = 1e-6 + F.softplus(self.b_rho)
                loss_kl += kl_loss_calculate(self.b_mu, b_sigma, self.prior_b_mu, prior_b_sigma)
        return F.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups), loss_kl


class LSTM(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers,
            dropout=0.5, bidirectional=False, return_last=True
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last
        self.D = 1
        if bidirectional is True:
            self.D = 2
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )

    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        if self.return_last:
            out = out[:, -1, :]
        return out


def sample_weights(w_mu, w_rho, b_mu, b_rho, bias):
    w_eps = w_mu.data.new(w_mu.size()).normal_()
    w_sigma = 1e-6 + F.softplus(w_rho)
    w = w_mu + 1 * w_sigma * w_eps
    if bias:
        b_eps = b_mu.data.new(b_mu.size()).normal_()
        b_sigma = 1e-6 + F.softplus(b_rho)
        b = b_mu + 1 * b_sigma * b_eps
    else:
        b = None
    return w, b


class BayesianConvNet(nn.Module):
    def __init__(
            self, n_timepoints, n_seqlen, n_classes, bayesian, mc_drop, device,
            # Conv layers
            n_filters_1=64, groups=1,
            filter_size_1=19, filter_stride_1=4,
            filter_size_2=11, filter_stride_2=4,
            pool_size_1=6, pool_stride_1=6,
            n_filters_2=64, filter_size_1x3=7,
            pool_size_2=4, pool_stride_2=4,
            # LSTM layers
            n_rnn_layers=1, dropout=0.5
    ):
        super().__init__()
        self.output_dim = n_classes
        self.n_seqlen = n_seqlen
        self.bayesian = bayesian
        self.mc_drop = mc_drop
        self.device = device

        if self.bayesian:
            self.conv1 = BayesConv(1, n_filters_1, filter_size_1, self.device, filter_stride_1, filter_size_1 // 2,
                                   bias=False, prior_w_sigma=-3)
        else:
            self.conv1 = nn.Conv1d(1, n_filters_1, filter_size_1, filter_stride_1, filter_size_1 // 2,
                                   bias=False)
        self.conv2 = nn.Conv1d(n_filters_1, n_filters_1, filter_size_2, filter_stride_2, filter_size_2 // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_1)
        self.pool1 = nn.MaxPool1d(pool_size_1, pool_stride_1)
        self.drop = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(n_filters_1, n_filters_1, filter_size_1x3, 1, "same", bias=False)
        self.conv4 = nn.Conv1d(n_filters_1, n_filters_1, filter_size_1x3, 1, "same", bias=False)
        self.conv5 = nn.Conv1d(n_filters_1, n_filters_1, filter_size_1x3, 1, "same", bias=False)
        self.pool2 = nn.MaxPool1d(pool_size_2, pool_stride_2)
        outlen_conv = n_timepoints // filter_stride_1 // filter_stride_2 // pool_stride_1 // pool_stride_2 * n_filters_2
        self.rnn_branch = LSTM(outlen_conv, n_filters_2, n_rnn_layers, dropout)
        self.linear = nn.Linear(n_filters_2, n_classes, bias=False)

    def forward(self, x, train=True):
        a, b, _, c, _ = x.size()
        x = x.view(a * b, 1, c)

        mean = torch.mean(x, dim=(0, 2), keepdim=True)
        std = torch.std(x, dim=(0, 2), keepdim=True)
        x = (x - mean) / std
        if self.bayesian:
            x, loss = self.conv1(x, sample=True)
        else:
            x = self.conv1(x)
            loss = torch.tensor(0).to(self.device)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pool1(x)
        if train:
            x = self.drop(x)
        else:
            if self.mc_drop:
                x = self.drop(x)

        x = self.conv3(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pool2(x)
        if train:
            x = self.drop(x)
        else:
            if self.mc_drop:
                x = self.drop(x)

        x = x.view(x.size(0), -1)
        x = x.view(-1, self.n_seqlen, x.size(-1))
        x = self.rnn_branch(x)
        if train:
            x = self.drop(x)
        else:
            if self.mc_drop:
                x = self.drop(x)

        x = self.linear(x)
        x = F.sigmoid(x)
        return x, loss

    def influence(self, x, n_samples=200):
        if self.bayesian or self.mc_drop:
            predictions = x.data.new(n_samples, x.shape[0], self.output_dim)
            for i in range(n_samples):
                y, _ = self.forward(x, train=False)
                y = y / torch.sum(y, dim=1, keepdim=True)
                predictions[i] = y
            std = torch.std(predictions, dim=0)
            return torch.mean(predictions, dim=0), std
        else:
            y, _ = self.forward(x, train=False)
            return y, torch.zeros(1).to(self.device)
