import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.utility import get_sub_seqs


class STEN:
    def __init__(self, seq_len=10, stride=1, epoch=5, batch_size=256, lr=1e-5,
                 hidden_dim=256, rep_dim=100, verbose=2, random_state=42,
                 random_strength=0.1, alpha=1, beta=1, device='cuda'):

        self.seq_len = seq_len  # 用傳入的超參數初始化
        self.stride = stride
        self.epochs = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.rep_dim = rep_dim
        self.verbose = verbose
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta
        self.device = device

        # 這些屬性是固定的
        self.num_rank = 10
        self.k = 5
        self.dl = 1e7
        self.epoch_steps = -1
        self.n_features = -1
        self.network = None

    def fit(self, x):
        self.n_features = x.shape[-1]
        seqs_lst = get_sub_seqs(x, seq_len=self.seq_len * self.num_rank, stride=self.stride)
        train_dataset = SeqDataset(seqs_lst, num_rank=self.num_rank, seed=self.random_state)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.train_load = train_loader
        self.network = GRUNetwork(input_dim=self.n_features, out_dim=self.num_rank,
                                  hidden_dim=self.hidden_dim, emb_dim=512, n_emb=128, num_gru=self.num_rank)
        self.Embnetwork = GRUNetworkEmb(input_dim=self.n_features, out_dim=self.num_rank,
                                        hidden_dim=self.hidden_dim, emb_dim=512, num_gru=self.num_rank)

        self.network.to(self.device)
        self.Embnetwork.to(self.device)

        random_criterion = torch.nn.MSELoss(reduction='mean')
        criterion = InterLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=1e-5)

        self.network.train()
        for i in range(self.epochs):
            t1 = time.time()
            train_loss_lst = []
            steps = 0

            with tqdm(total=len(train_loader), desc=f'epoch {i + 1:3d}/{self.epochs}', ncols=150) as pbar:
                for idx, (batch_data1, batch_data2, batch_label) in enumerate(train_loader):
                    seq_batch_data_lst = []
                    seq_batch_data_lst_random = []

                    seq_batch_data = batch_data1.view(self.batch_size, self.num_rank, self.seq_len, self.n_features)
                    seq_batch_data_random = batch_data2.view(self.batch_size, self.num_rank, self.seq_len, self.n_features).float().to(self.device)

                    # shuffle
                    for n in range(self.batch_size):
                        shuffle_label = batch_label[n] - 1
                        seq_batch_data_shuffle = seq_batch_data[n, shuffle_label, :, :]
                        seq_batch_data_lst.append(seq_batch_data_shuffle)
                    seq_batch_shuffle_data = torch.stack(seq_batch_data_lst)
                    seq_batch_shuffle_data = seq_batch_shuffle_data.float().to(self.device)

                    seq_batch_data = seq_batch_data.float().to(self.device)

                    for n in range(self.batch_size):
                        shuffle_label = batch_label[n] - 1
                        seq_batch_data_shuffle_random = seq_batch_data_random[n, shuffle_label, :, :]
                        seq_batch_data_lst_random.append(seq_batch_data_shuffle_random)
                    seq_batch_shuffle_data_random = torch.stack(seq_batch_data_lst_random)
                    seq_batch_shuffle_data_random = seq_batch_shuffle_data_random.float().to(self.device)

                    seq_batch_data_random = seq_batch_data_random.float().to(self.device)

                    seq_batch_label = batch_label.float().to(self.device)

                    pred, pred_dis = self.network(seq_batch_data, seq_batch_shuffle_data)
                    _, pred_dis_random = self.network(seq_batch_data_random, seq_batch_shuffle_data_random)

                    pred_dis = pred_dis.detach()
                    pred_dis_random = pred_dis_random.detach()
                    xy_dis = (F.normalize(pred_dis, p=1, dim=1) * F.normalize(pred_dis_random, p=1, dim=1)).sum(dim=1)

                    pred_target = self.Embnetwork(seq_batch_data)
                    pred_target_random = self.Embnetwork(seq_batch_data_random)
                    pred_target = pred_target.detach()
                    pred_target_random = pred_target_random.detach()
                    x_y_dis = (F.normalize(pred_target, p=1, dim=1) * F.normalize(pred_target_random, p=1, dim=1)).sum(dim=1)

                    # rank loss
                    loss = criterion(pred, seq_batch_label)
                    # distance loss
                    dis_loss_raw = random_criterion(xy_dis, x_y_dis)

                    dis_loss = self.dl * dis_loss_raw
                    loss = loss + self.alpha * dis_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.item()
                    steps += 1
                    train_loss_lst.append(batch_loss)

                    pbar.set_postfix(loss=f'{np.mean(batch_loss):.6f}')
                    pbar.update(1)

                    if self.epoch_steps != -1:
                        if idx > self.epoch_steps:
                            break

            train_loss_lst = np.array(train_loss_lst)
            train_loss_avg = np.average(train_loss_lst, axis=0)
            epoch_loss_str = f'{np.mean(train_loss_avg):.6f}'

            t = time.time() - t1
            print(f'epoch {i + 1:3d}/{self.epochs}: '
                  f'loss={epoch_loss_str} | '
                  f'time={t: .1f}s | '
                  f'steps={steps}')

            for name, param in self.network.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print("Gradient contains NaN or inf values for parameter: ", name)

        return

    def decision_function(self, x):
        length = len(x)
        seqs = get_sub_seqs(x, seq_len=self.seq_len * self.num_rank, stride=1)
        random_criterion = torch.nn.MSELoss(reduction='none')
        criterion = InterLoss(reduction='none')
        ensemble_score_lst = []

        test_dataset = TestDataset(seqs, num_rank=self.num_rank, seed=self.random_state)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.network.eval()
        with torch.no_grad():
            score_lst = []
            with tqdm(total=len(test_loader), desc=f'testing len={self.seq_len}', ncols=150) as pbar:
                for batch_data1, batch_data2, batch_label in test_loader:
                    seq_batch_data_lst = []
                    dis_loss_arr = []
                    rank_loss_arr = []
                    batch_size = batch_data1.size(0)
                    seq_batch_data = batch_data1.view(batch_size, self.num_rank, self.seq_len, self.n_features)

                    for m in range(batch_size):
                        shuffle_label = batch_label[m] - 1
                        seq_batch_data_shuffle = seq_batch_data[m, shuffle_label, :, :]
                        seq_batch_data_lst.append(seq_batch_data_shuffle)
                    seq_batch_shuffle_data = torch.stack(seq_batch_data_lst)
                    seq_batch_shuffle_data = seq_batch_shuffle_data.float().to(self.device)

                    seq_batch_data = seq_batch_data.float().to(self.device)

                    for a in range(5):
                        seq_batch_data_random = batch_data2[a].view(batch_size, self.num_rank, self.seq_len, self.n_features)
                        seq_batch_data_lst_random = []
                        for r in range(batch_size):
                            shuffle_label = batch_label[r] - 1
                            seq_batch_data_shuffle_random = seq_batch_data_random[r, shuffle_label, :, :]
                            seq_batch_data_lst_random.append(seq_batch_data_shuffle_random)
                        seq_batch_shuffle_data_random = torch.stack(seq_batch_data_lst_random)
                        seq_batch_shuffle_data_random = seq_batch_shuffle_data_random.float().to(self.device)

                        seq_batch_data_random = seq_batch_data_random.float().to(self.device)

                        seq_batch_label = batch_label.float()
                        seq_batch_label = batch_label.reshape([seq_batch_label.shape[0], -1]).float().to(self.device)

                        pred, pred_dis = self.network(seq_batch_data, seq_batch_shuffle_data)
                        _, pred_dis_random = self.network(seq_batch_data_random, seq_batch_shuffle_data_random)

                        pred_dis = pred_dis.detach()
                        pred_dis_random = pred_dis_random.detach()
                        xy_dis = (F.normalize(pred_dis, p=1, dim=1) * F.normalize(pred_dis_random, p=1, dim=1)).sum(dim=1)

                        pred_target = self.Embnetwork(seq_batch_data)
                        pred_target_random = self.Embnetwork(seq_batch_data_random)
                        pred_target = pred_target.detach()
                        pred_target_random = pred_target_random.detach()
                        x_y_dis = (F.normalize(pred_target, p=1, dim=1) * F.normalize(pred_target_random, p=1, dim=1)).sum(dim=1)

                        dis_loss = random_criterion(xy_dis, x_y_dis).cpu().numpy()
                        dis_loss_arr.append(dis_loss)

                        pred_s = F.softmax(pred, dim=1)
                        label_s = F.softmax(seq_batch_label, dim=1)
                        item_loss = torch.abs(pred_s - label_s).cpu()

                        rank_loss = criterion(pred, seq_batch_label)
                        rank_loss = rank_loss.flatten()
                        loss = rank_loss.cpu()
                        reshape_loss = np.zeros((batch_size, 10))
                        reshape_loss[:] = loss[:, np.newaxis]
                        rank_loss = item_loss / reshape_loss
                        rank_loss_arr.append(rank_loss.numpy())

                    dis_loss = np.average(dis_loss_arr, axis=0)
                    dis_loss_reshape = np.zeros((batch_size, 10))
                    dis_loss_reshape[:] = dis_loss[:, np.newaxis]
                    rank_loss = np.average(rank_loss_arr, axis=0)

                    anomaly_score = rank_loss + self.beta * dis_loss_reshape
                    score_lst.append(anomaly_score.data)

                    pbar.update(1)

        score_lst = np.concatenate(score_lst)
        score_lst = self.get_score(score_lst, length)

        _max_, _min_ = np.max(score_lst), np.min(score_lst)
        score_lst = (score_lst - _min_) / (_max_ - _min_)
        ensemble_score_lst.append(score_lst)

        ensemble_score_lst = np.array(ensemble_score_lst)
        scores = np.average(ensemble_score_lst, axis=0)
        padding = np.zeros(self.seq_len - 1)

        assert padding.shape[0] + scores.shape[0] == x.shape[0]
        scores = np.hstack((padding, scores))

        return scores

    def get_score(self, input, length):
        S_lst = [[] for _ in range(length)]

        for i in range(len(input)):
            for j in range(10):
                S_lst[i + (j + 1) * self.seq_len - 1].append(input[i][j])
        New_lst = [lst for lst in S_lst if lst]

        avg_lst = []
        for seqs in New_lst:
            avg = sum(seqs) / len(seqs)
            avg_lst.append(avg)

        scores = avg_lst

        return scores
