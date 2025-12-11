# ltkd_adapter_cifar.py
# CIFAR(10/100) 长尾分类 —— 多教师离线蒸馏 + 参考熵触发适配器（新理论）
# 依赖：PyTorch >= 2.0，torchvision，numpy，scikit-learn，tensorboardX(可选)
# 运行示例：
# D:\appp\CUT-KD\envs\Scripts\Activate.ps1  
#   python D:\appp\CUT-KD\cut-kd.py --dataset cifar10   --datapath D:\appp\CUT-KD\CIFAR\data   --lt_dir   D:\appp\CUT-KD\CIFAR\data\cifar-10-LT-10   --out_dir  D:\appp\CUT-KD\results --epochs 30 --batch_size 256 --lr 0.1   --eta 0.4 --lambda_max 0.8 --rho 0.25 --tau_pct 80

import os, argparse, tqdm, numpy as np, time, pathlib, math
from collections import defaultdict
from typing import Dict, Tuple

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Sampler, Dataset
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

try:
    autocast = torch.amp.autocast
except AttributeError:
    from torch.cuda.amp import autocast

try:
    from tensorboardX import SummaryWriter
except Exception:
    from torch.utils.tensorboard import SummaryWriter

# -------------------- 公用：度量与工具（保持与原版一致） --------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.cnt = 0.
    def update(self, v, n): self.sum += v * n; self.cnt += n
    @property
    def avg(self): return self.sum / max(1, self.cnt)

@torch.no_grad()
def topk(logit, tgt, k=1):
    _, pred = logit.topk(k, 1, True, True)
    return pred.eq(tgt.view(-1, 1)).float().sum().mul_(100. / tgt.size(0))

def seg_split(labels: np.ndarray):
    cnt = np.bincount(labels); idx = np.argsort(cnt)[::-1]
    cum = np.cumsum(cnt[idx]) / cnt.sum()
    head = idx[cum <= 0.5]; mid = idx[(cum > 0.5) & (cum <= 0.9)]; tail = idx[cum > 0.9]
    return dict(head=head, mid=mid, tail=tail)

def fmt(x: float) -> str:
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.4f}"

# -------------------- 数据（保持与原版一致） --------------------
class TwoViewsDataset(Dataset):
    def __init__(self, base_dataset, indices, T1, T2):
        self.base = base_dataset
        self.indices = indices
        self.T1 = T1
        self.T2 = T2
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        img, lbl = self.base[idx]
        return self.T1(img), self.T2(img), lbl

class DifficultySampler(Sampler):
    """λ_RS 融合 (尾类×难度×熵) 概率，与 uniform 混合；与原实现一致。"""
    def __init__(self, labels: np.ndarray, lambda_rs=0.5):
        self.labels = labels
        self.N = len(labels)
        cls_cnt = np.bincount(labels)
        inv_freq = 1. / np.maximum(cls_cnt[labels], 1)
        self.base = inv_freq / inv_freq.mean()
        self.lambda_rs = lambda_rs
        self.uniform = np.ones(self.N) / self.N
        self.p = self.uniform.copy()
    def update(self, ce_hist, entropy):
        q = self.base * ce_hist * (1. + entropy)
        s = q.sum()
        if s <= 0 or np.isnan(s):
            self.p = self.uniform.copy(); return
        q /= s
        self.p = (1 - self.lambda_rs) * self.uniform + self.lambda_rs * q
    def __iter__(self): return iter(np.random.choice(self.N, self.N, p=self.p))
    def __len__(self):  return self.N

class SubsetWithIndex(Dataset):
    """包装 Subset：返回 (img, lbl, global_idx)。"""
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        gi = self.indices[i]
        img, lbl = self.base[gi]
        return img, lbl, gi

def cifar_loaders(cfg):
    T_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    T_test  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])

    Data = datasets.CIFAR10 if cfg['dataset'] == 'cifar10' else datasets.CIFAR100

    # 长尾索引
    if cfg.get('lt_dir'):
        lt_root = cfg['lt_dir']
    else:
        lt_root = os.path.join(cfg['datapath'],
            f"{'cifar-10' if cfg['dataset']=='cifar10' else 'cifar-100'}-LT-10")
    idx_file = os.path.join(lt_root, 'indices_train_lt.txt')
    if not os.path.isfile(idx_file):
        raise FileNotFoundError(f'找不到长尾索引 {idx_file}')
    with open(idx_file) as f:
        idx = [int(i) for i in f.read().split()]

    full_train      = Data(cfg['datapath'], True,  download=True, transform=T_train)
    train_set       = SubsetWithIndex(full_train, idx)  # << 仅在这里多返回 global_idx
    val_set         = Data(cfg['datapath'], False, download=True, transform=T_test)

    # labels（与原一致）
    full_labels = np.array(full_train.targets)
    labels = full_labels[idx]

    u_loader = DataLoader(train_set, cfg['batch_size'],
                          sampler=RandomSampler(train_set),
                          num_workers=6, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, cfg['batch_size'],
                            shuffle=False, num_workers=6, pin_memory=True)
    return u_loader, val_loader, labels, np.array(idx, dtype=np.int64), len(full_train)
# -------------------- 学生网络（共享骨干 + 主头 + 适配器） --------------------
class CifarResNet18(nn.Module):
    """ResNet-18 (conv3-s1, 无 maxpool) 输出 512-d；与原骨干一致。"""
    def __init__(self):
        super().__init__()
        from torchvision.models.resnet import BasicBlock
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._block(BasicBlock, 64, 2)
        self.layer2 = self._block(BasicBlock, 128, 2, 2)
        self.layer3 = self._block(BasicBlock, 256, 2, 2)
        self.layer4 = self._block(BasicBlock, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
    def _block(self, blk, planes, blocks, stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*blk.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.inplanes, planes*blk.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes*blk.expansion)
            )
        layers=[blk(self.inplanes, planes, stride, downsample)]
        self.inplanes=planes*blk.expansion
        for _ in range(1,blocks):
            layers.append(blk(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.avgpool(x).flatten(1)  # [B,512]

class HeadMLP(nn.Module):
    def __init__(self, d, ncls):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, ncls)
    def forward(self, z):  # logits
        return self.fc2(F.relu(self.fc1(z), inplace=True))

class Student(nn.Module):
    """
    S0: 主头；C: 适配器头（可升级为 S1，同结构）
    predict(): 返回 log-prob（便于与原 evaluate 对接）
    """
    def __init__(self, ncls=10, rho=0.2):
        super().__init__()
        self.backbone = CifarResNet18()
        self.s0 = HeadMLP(self.backbone.out_dim, ncls)  # 主学生
        self.adp = HeadMLP(self.backbone.out_dim, ncls)  # 适配器 C
        self.rho = rho
        self.tau = float('inf')
    @staticmethod
    def _softmax_logit(logits):  # 返回 (p, logp)
        p = F.softmax(logits, -1)
        return p, (p+1e-12).log()
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        p0 = F.softmax(self.s0(z), -1)
        d = -(p0 * (p0+1e-12).log()).sum(-1)
        trigger = (d > self.tau).float().unsqueeze(-1)
        
        pC = F.softmax(self.adp(z), -1)
        pS = torch.where(trigger.bool(), (1-self.rho)*p0 + self.rho*pC, p0)
        return (pS+1e-12).log()

# -------------------- 离线教师缓存与融合（新理论） --------------------
class TeacherCache:
    """
    以 numpy/pt 载入离线教师 logits：形状 [N_train_full, C]，对应原始训练集全索引。
    训练实际使用长尾子集 indices 时按 global_idx 切片。
    """
    def __init__(self, paths, n_full, ncls, temperature=3.0, delta_s=0.05, utility_scale=1.0, device='cuda'):
        """
        paths: [path_t0, path_t1, path_t2] (可为 None 表示不存在某教师)
        """
        self.device = torch.device(device)
        self.T = float(temperature)
        self.delta_s = float(delta_s)
        self.k = 3
        self.n_full = n_full
        self.ncls = ncls
        self.present = [False, False, False]
        self.pruned = [False, False, False]
        self.scales = [1.0, 1.0, 1.0]
        self.utility_scale = float(utility_scale)
        self.Z = [None, None, None]  # CPU tensors
        for k, p in enumerate(paths):
            if p is None: 
                continue
            arr = self._load_array(p)
            assert arr.shape[0] == n_full, f"Teacher {k} N mismatch: {arr.shape[0]} vs {n_full}"
            assert arr.shape[1] == ncls, f"Teacher {k} C mismatch: {arr.shape[1]} vs {ncls}"
            self.Z[k] = torch.from_numpy(arr.astype(np.float32)).to(self.device)
            self.present[k] = True  # ✅ 必须设置
            self.scales[k] = 1.0    # 初始化为 1.0

    def _load_array(self, path):
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.pt') or path.endswith('.pth'):
            t = torch.load(path, map_location='cpu')
            if isinstance(t, torch.Tensor): return t.numpy()
            elif isinstance(t, dict) and 'logits' in t: return t['logits']
            else: raise ValueError(f"Unknown PT format: {path}")
        else:
            raise ValueError(f"Unsupported teacher file: {path}")

    def compute_scales(self, indices_subset):
        """
        s_k = sigma(z0) / sigma(zk)，若 |s_k-1|<δ 则置 1；在训练开始前一次性计算。
        """
        if not self.present[0]:
            # 若无 T0，则所有 s_k=1
            return
        idx = torch.as_tensor(indices_subset, device=self.device, dtype=torch.long)
        z0 = self.Z[0].index_select(0, idx)  # [N_sub, C]
        sig0 = z0.float().std().item() + 1e-12
        for k in range(1,3):
            if self.present[k]:
                zk = self.Z[k].index_select(0, idx)
                sigk = zk.float().std().item() + 1e-12
                sk = sig0 / sigk
                if abs(sk - 1.0) < self.delta_s:
                    sk = 1.0
                self.scales[k] = float(sk)

    def prune_after_warmup(self, labels_subset, indices_subset, seg_map, r_th=0.15, nmin_ratio=0.005):
        """
        依据式(11)：r_k 统计在覆盖域内“最自信”的占比；低于阈值则剪枝。
        """
        N_sub = len(indices_subset)
        N_min = max(100, int(nmin_ratio * N_sub + 0.5))
        # 覆盖矩阵 M（行 T0,T1,T2；列 H,M,T）
        M = torch.tensor([[1,1,1],[0,1,1],[0,0,1]], dtype=torch.bool)
        # 类→段映射
        C = int(labels_subset.max()) + 1
        seg_label = torch.full((C,), 0, dtype=torch.long)  # 0:H,1:M,2:T
        seg_label[torch.tensor(seg_map['head'])] = 0
        seg_label[torch.tensor(seg_map['mid'])]  = 1
        seg_label[torch.tensor(seg_map['tail'])] = 2

        # 预取所有教师在子集上的 softmax(T) 概率（经缩放）
        P = []
        idx = torch.as_tensor(indices_subset, device=self.device, dtype=torch.long)
        for k in range(3):
            if self.present[k]:
                Zk = self.Z[k].index_select(0, idx) * self.scales[k]     # [N,C] on device
                Pk = F.softmax(Zk / self.T, dim=-1).cpu()                # 搬回 CPU，后面大量取标量更快
                P.append(Pk)
            else:
                P.append(None)

        r_num = [0,0,0]; r_den=[0,0,0]
        for i in range(N_sub):
            y = int(labels_subset[i])
            col = int(seg_label[y])  # 段
            active = []
            for k in range(3):
                if (not self.present[k]) or self.pruned[k]: continue
                if M[k, col]:
                    active.append(k)
            if len(active)==0: continue
            # 该样本的教师实用度 u_k = p_k(y|x)
            uvals = [P[k][i, y].item() for k in active]
            # 增计分母
            for k in active:
                r_den[k] += 1
            # 最自信教师
            k_star = active[int(np.argmax(uvals))]
            r_num[k_star] += 1

        for k in range(3):
            if r_den[k] < N_min:  # 证据不足，不剪
                continue
            rk = (r_num[k] / max(1, r_den[k]))
            if rk < r_th:
                self.pruned[k] = True
                print(f">> Prune teacher T{k}: r={rk:.4f} < {r_th}")

    def fused_target(self, indices, labels, t, eta):
        """
        对一个 batch（global indices + labels）：返回
         - has_kd: [B] bool（是否存在覆盖教师）
         - p_star: [B,C]（仅对 has_kd==1 的样本有效；其余值未用）
        实现：阶段门控 + 覆盖矩阵 + 实用度加权几何平均（或等价的 logit 混合）。
        """
        B = len(indices)
        device = self.device
        stage_flags = [(t < eta), (t >= eta), (t >= eta)]
        M = torch.tensor([[1,1,1],[0,1,1],[0,0,1]], dtype=torch.bool, device=device)

        C = self.ncls
        seg_col = self.seg_label[labels]
        
        has_kd = torch.zeros(B, dtype=torch.bool, device=device)
        z_mix = torch.zeros(B, C, dtype=torch.float32, device=device)
        
        idx = indices.to(self.device).long() if isinstance(indices, torch.Tensor) \
              else torch.as_tensor(indices, device=self.device, dtype=torch.long)
        
        # 仅构建 Z_dict（删除冗余的 Z）
        Z_dict = {}
        for k in range(3):
            if self.present[k] and (not self.pruned[k]) and stage_flags[k]:
                z_k = self.Z[k].index_select(0, idx) * self.scales[k]
                Z_dict[k] = z_k
        
        for i in range(B):
            col = int(seg_col[i])
            active = []
            for k in range(3):
                if k not in Z_dict:
                    continue
                if M[k, col]:
                    active.append(k)
            
            if len(active) == 0:
                continue
            
            has_kd[i] = True
            y = int(labels[i].item())
            
            z_active = torch.stack([Z_dict[k][i] for k in active])
            p_active = F.softmax(z_active / self.T, dim=-1)
            u_vals = p_active[:, y]
            
            alpha = F.softmax(self.utility_scale * u_vals, dim=0)
            zstar = (alpha.unsqueeze(-1) * z_active / self.T).sum(0)
            z_mix[i] = zstar
        
        p_star = F.softmax(z_mix, dim=-1)
        return has_kd, p_star

# -------------------- 评测（保持与原版一致） --------------------
@torch.no_grad()
def evaluate(model: Student, loader, seg_map, tb, ep, amp=False):
    model.eval()
    top1 = AverageMeter()
    per_cls = defaultdict(list)
    all_logits, all_labels = [], []

    for batch in loader:
        if len(batch)==2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        x, y = x.cuda(), y.cuda()
        with autocast('cuda', enabled=amp):
            logits = model.predict(x)  # log-prob
        top1.update(topk(logits, y).item(), x.size(0))
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
        preds = logits.argmax(1).cpu()
        for t, p in zip(y.cpu(), preds):
            per_cls[t.item()].append(int(t == p))

    def seg_metrics(ids):
        mask = torch.isin(torch.cat(all_labels), torch.tensor(ids))
        if mask.sum() == 0:
            return dict(acc=float('nan'), auc=None, gmean=float('nan'), f1=float('nan'))
        y_seg = torch.cat(all_labels)[mask]
        p_seg = torch.cat(all_logits)[mask]
        acc = (p_seg.argmax(1) == y_seg).float().mean().item() * 100
        try:
            auc = roc_auc_score(
                y_seg.numpy(), F.softmax(p_seg, -1).numpy(),
                multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')
        f1 = f1_score(y_seg.numpy(), p_seg.argmax(1).numpy(), average='macro')
        cm = confusion_matrix(y_seg, p_seg.argmax(1), labels=ids)
        rec = np.diag(cm) / cm.sum(1).clip(min=1)
        gmean = float(np.exp(np.log(np.clip(rec,1e-12,1)).mean()))
        return dict(acc=acc, auc=auc, gmean=gmean, f1=f1)

    head_m = seg_metrics(seg_map['head'])
    mid_m  = seg_metrics(seg_map['mid'])
    tail_m = seg_metrics(seg_map['tail'])

    Y = torch.cat(all_labels); P = torch.cat(all_logits)
    try:
        auc_all = roc_auc_score(
            Y.numpy(), F.softmax(P, -1).numpy(), multi_class='ovr', average='macro')
    except ValueError:
        auc_all = float('nan')
    f1_all  = f1_score(Y.numpy(), P.argmax(1).numpy(), average='macro')
    cm_all  = confusion_matrix(Y.numpy(), P.argmax(1).numpy(), labels=np.arange(P.size(1)))
    rec_all = np.diag(cm_all) / cm_all.sum(1).clip(min=1)
    gmean_all = float(np.exp(np.log(np.clip(rec_all,1e-12,1)).mean()))
    metr = dict(
        acc_all = top1.avg,
        auc_all = auc_all,
        gmean_all = gmean_all,
        f1_all  = f1_all,
        head_acc=head_m['acc'], head_auc=head_m['auc'], head_gmean=head_m['gmean'], head_f1=head_m['f1'],
        mid_acc =mid_m['acc'],  mid_auc =mid_m['auc'],  mid_gmean=mid_m['gmean'],  mid_f1=mid_m['f1'],
        tail_acc=tail_m['acc'], tail_auc=tail_m['auc'], tail_gmean=tail_m['gmean'], tail_f1=tail_m['f1'],
    )
    metr['auc'] = roc_auc_score(
        Y.numpy(), F.softmax(P, -1).numpy(), multi_class='ovr', average='macro')
    metr['f1']  = f1_score(
        Y.numpy(), P.argmax(1).numpy(), average='macro')
    if tb is not None:
        for k, v in metr.items():
            if v is not None and np.isfinite(v):
                tb.add_scalar('val/' + k, v, ep)
    return metr

# -------------------- 输出写入（保持与原版一致） --------------------
def write_csv(cfg, metr, file_path='results.csv'):
    path = pathlib.Path(file_path)
    if not path.exists():
        path.write_text(
            "method,seed,epoch,acc,auc,gmean,f1,"
            "head_acc,head_auc,head_gmean,head_f1,"
            "mid_acc,mid_auc,mid_gmean,mid_f1,"
            "tail_acc,tail_auc,tail_gmean,tail_f1,file\n")
    with open(path, 'a', newline='', encoding='utf-8') as f:
        f.write(
            f"{cfg.get('method','ltkd_adp')},{cfg['seed']},{cfg['epochs']},"
            f"{fmt(metr['acc_all'])},{fmt(metr['auc_all'])},{fmt(metr['gmean_all'])},{fmt(metr['f1_all'])},"
            f"{fmt(metr['head_acc'])},{fmt(metr['head_auc'])},{fmt(metr['head_gmean'])},{fmt(metr['head_f1'])},"
            f"{fmt(metr['mid_acc'])},{fmt(metr['mid_auc'])},{fmt(metr['mid_gmean'])},{fmt(metr['mid_f1'])},"
            f"{fmt(metr['tail_acc'])},{fmt(metr['tail_auc'])},{fmt(metr['tail_gmean'])},{fmt(metr['tail_f1'])},final.pth\n")

def write_result(cfg, metr, file_path='results.md'):
    path = pathlib.Path(file_path)
    if not path.exists():
        path.write_text(
            "| method | seed | epoch | acc | auc | gmean | f1 | "
            "head_acc | head_auc | head_gmean | head_f1 | "
            "mid_acc | mid_auc | mid_gmean | mid_f1 | "
            "tail_acc | tail_auc | tail_gmean | tail_f1 | file |\n"
            "|--------|------|-------|-----|-----|----|"
            "---------|---------|--------|"
            "--------|---------|--------|"
            "---------|---------|--------|------|\n")
    with open(path, 'a', encoding='utf-8') as f:
        f.write(
            f"| {cfg.get('method','ltkd_adp')} | {cfg['seed']} | {cfg['epochs']} | "
            f"{fmt(metr['acc_all'])} | {fmt(metr['auc_all'])} | {fmt(metr['gmean_all'])} | {fmt(metr['f1_all'])} | "
            f"{fmt(metr['head_acc'])} | {fmt(metr['head_auc'])} | {fmt(metr['head_gmean'])} | {fmt(metr['head_f1'])} | "
            f"{fmt(metr['mid_acc'])} | {fmt(metr['mid_auc'])} | {fmt(metr['mid_gmean'])} | {fmt(metr['mid_f1'])} | "
            f"{fmt(metr['tail_acc'])} | {fmt(metr['tail_auc'])} | {fmt(metr['tail_gmean'])} | {fmt(metr['tail_f1'])} | final.pth |\n")

# -------------------- 训练例程（按新理论） --------------------
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train(cfg):
    device = 'cuda'
    torch.cuda.set_device(cfg.get('gpu', 0))
    torch.backends.cudnn.benchmark = True
    set_seed(cfg['seed'])

    output_dir = pathlib.Path(cfg.get("out_dir") or ".")
    output_dir.mkdir(parents=True, exist_ok=True)


    u_loader, val_loader, labels, indices_subset, n_full = cifar_loaders(cfg)
    seg_map = seg_split(labels)
    ncls = cfg['num_classes']

    seg_label = torch.full((ncls,), 0, dtype=torch.long)
    seg_label[torch.tensor(seg_map['head'])] = 0
    seg_label[torch.tensor(seg_map['mid'])]  = 1
    seg_label[torch.tensor(seg_map['tail'])] = 2

    model = Student(ncls, rho=cfg['rho']).cuda()
    opt = torch.optim.SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=5e-4)
    try:
        scaler = torch.amp.GradScaler(enabled=cfg['amp'])
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'])
    tb = SummaryWriter(comment=cfg['dataset'])
    global_step = 0

    teachers = TeacherCache(
        paths=[cfg.get('t0_path'), cfg.get('t1_path'), cfg.get('t2_path')],
        n_full=n_full, ncls=ncls, temperature=cfg['temperature'],
        delta_s=cfg['delta_s'], utility_scale=cfg['utility_scale'], device=device  # 新增 utility_scale
    )
    teachers.seg_label = seg_label.cuda()
    teachers.compute_scales(indices_subset)

    cls_cnt = np.bincount(labels, minlength=ncls)
    w_cb = np.array([(1 - cfg['beta']) / (1 - cfg['beta'] ** max(1, c)) for c in cls_cnt], dtype=np.float32)
    w_cb = torch.from_numpy(w_cb).cuda()

    ent_epoch = np.zeros(len(indices_subset), dtype=np.float32)
    tau = float('inf'); tau_ready = False

    best_tail = 0.0

    for ep in range(cfg['epochs']):
        model.train()
        meter = AverageMeter()
        pbar = tqdm.tqdm(u_loader, desc=f'E{ep}')
        tic = time.time()
        
        ent_epoch.fill(0.0)
        count_epoch = np.zeros_like(ent_epoch)
        
        if ep == 0:
            inv_map = torch.full((n_full,), -1, dtype=torch.long, device='cpu')
            inv_map[torch.from_numpy(indices_subset)] = torch.arange(len(indices_subset))
        
        for batch in pbar:
            x, y, gi = batch
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            gi = gi.cuda(non_blocking=True)
            
            t = ep / max(1, cfg['epochs'] - 1)
            t = float(np.clip(t, 0.0, 1.0))
            
            with autocast('cuda', enabled=cfg['amp']):
                z = model.backbone(x)
                logit_s0 = model.s0(z)
                logit_ad = model.adp(z)
                
                p0 = F.softmax(logit_s0, -1)
                has_kd, p_star = teachers.fused_target(gi, y, t, cfg['eta'])
                p_ref = torch.where(has_kd.unsqueeze(-1), p_star, p0)
                
                d = -(p_ref * (p_ref+1e-12).log()).sum(-1)
                idx_pos = inv_map[gi.cpu()]
                assert (idx_pos >= 0).all(), "Invalid global index"
                idx_np = idx_pos.cpu().numpy()
                ent_epoch[idx_np] += d.detach().cpu().numpy()
                count_epoch[idx_np] += 1
                
                tau_tensor = torch.full_like(d, fill_value=model.tau if tau_ready else float('inf'))
                trig = (d > tau_tensor).float().unsqueeze(-1)
                
                pC = F.softmax(logit_ad, -1)
                pS = torch.where(trig.bool(), (1-cfg['rho'])*p0 + cfg['rho']*pC, p0)
                logpS = (pS + 1e-12).log()
                
                lam_t = cfg['lambda_max'] * 0.5 * (1 - math.cos(math.pi * t))
                lam_mask = has_kd.float()
                kd_loss = -(p_star * logpS).sum(-1) * lam_mask * (cfg['temperature']**2)
                
                py = torch.gather(pS, 1, y.view(-1,1)).squeeze(1)
                wy = w_cb[y]
                cb = - wy * ((1 - py).clamp_min(1e-12) ** cfg['gamma']) * (py + 1e-12).log()
                
                L = lam_t * kd_loss + (1 - lam_t * lam_mask) * cb
                loss = L.mean()
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
            meter.update(loss.item(), x.size(0))
            pbar.set_postfix(loss=f'{meter.avg:.3f}')
            
            if tb:
                tb.add_scalar('train/loss', loss.item(), global_step)
                tb.add_scalar('train/lam_t', lam_t, global_step)
                tb.add_scalar('train/p_act_batch', trig.float().mean().item(), global_step)
            global_step += 1
        
        valid_mask = count_epoch > 0
        ent_epoch[valid_mask] /= count_epoch[valid_mask]
        
        q = float(np.quantile(ent_epoch[valid_mask], cfg['tau_pct'] / 100.0))
        if not tau_ready:
            tau = q
            tau_ready = True
        else:
            tau = cfg['mu'] * tau + (1 - cfg['mu']) * q
        model.tau = float(tau)
        if tb:
            tb.add_scalar('train/tau', model.tau, ep)
        
        if ep == 0 and cfg.get('do_teacher_pruning', True):
            teachers.prune_after_warmup(
                labels_subset=labels,
                indices_subset=indices_subset,
                seg_map=seg_map,
                r_th=cfg['r_th'],
                nmin_ratio=cfg['nmin_ratio']
            )
        
        metr = evaluate(model, val_loader, seg_map, tb, ep, amp=cfg['amp'])
        write_csv(cfg | {"epochs": ep + 1}, metr, output_dir / "results.csv")
        write_result(cfg | {"epochs": ep + 1}, metr, output_dir / "results.md")
        

        tail_acc = metr['tail_acc'] if np.isfinite(metr['tail_acc']) else 0.0
        if tail_acc > best_tail + 1e-6:
            best_tail = tail_acc
            torch.save(model.state_dict(), output_dir / "best_tail.pth")
        
        toc = time.time() - tic
        print(
            f"Epoch {ep+1}/{cfg['epochs']} | "
            f"train_loss {meter.avg:.4f} | "
            f"val {fmt(metr['acc_all'])}/{fmt(metr['tail_acc'])} | "
            f"τ {model.tau:.4f} | t {toc:.1f}s"
        )
    
    torch.save(model.state_dict(), output_dir / "final.pth")
    write_result(cfg, metr, file_path=output_dir / "results.md")
    write_csv(cfg, metr, file_path=output_dir / "results.csv")
    tb.close()
# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据 & 训练
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('--datapath', default='./data')
    parser.add_argument('--lt_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--amp', type=lambda s: str(s).lower() in ['true','1','yes','y'], default=True)
    parser.add_argument('--lambda_rs', type=float, default=0.6)   # 仅用于原 DifficultySampler 的维护

    # 教师缓存
    parser.add_argument('--t0_path', type=str, default=None, help='T0 logits path [.npy/.pt]')
    parser.add_argument('--t1_path', type=str, default=None, help='T1 logits path [.npy/.pt]')
    parser.add_argument('--t2_path', type=str, default=None, help='T2 logits path [.npy/.pt]')
    parser.add_argument('--temperature', type=float, default=4.0)   # T
    parser.add_argument('--delta_s', type=float, default=0.05)      # |s_k-1|<δ_s → s_k=1
    parser.add_argument('--utility_scale', type=float, default=1.0) # 论文 Eq.7 中的 a 参数
    parser.add_argument('--eta', type=float, default=0.5)           # 阶段分界
    parser.add_argument('--r_th', type=float, default=0.15)         # 剪枝阈值
    parser.add_argument('--nmin_ratio', type=float, default=0.005)  # 证据下限占比

    # 适配器与阈值
    parser.add_argument('--rho', type=float, default=0.2)           # 适配器混合系数
    parser.add_argument('--tau_pct', type=float, default=80.0)      # 分位数
    parser.add_argument('--mu', type=float, default=0.95)           # τ 的 EMA 系数

    # 损失
    parser.add_argument('--lambda_max', type=float, default=0.7)    # KD 最大权重
    parser.add_argument('--beta', type=float, default=0.999)        # CB beta
    parser.add_argument('--gamma', type=float, default=2.0)         # Focal gamma

    parser.add_argument('--out_dir', type=str, default=None, help='结果保存目录（可选）')
    
    args = parser.parse_args()
    cfg = {
        'dataset': args.dataset,
        'datapath': args.datapath,
        'lt_dir': args.lt_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_classes': args.num_classes,
        'seed': args.seed,
        'gpu': args.gpu,
        'amp': bool(args.amp),
        'lambda_rs': args.lambda_rs,
        't0_path': args.t0_path, 't1_path': args.t1_path, 't2_path': args.t2_path,
        'temperature': args.temperature,
        'delta_s': args.delta_s,
        'utility_scale': args.utility_scale,
        'eta': args.eta,
        'r_th': args.r_th,
        'nmin_ratio': args.nmin_ratio,
        'rho': args.rho,
        'tau_pct': args.tau_pct,
        'mu': args.mu,
        'lambda_max': args.lambda_max,
        'beta': args.beta,
        'gamma': args.gamma,
        'method': 'ltkd_adp',
        'out_dir': args.out_dir,
    }
    train(cfg)


