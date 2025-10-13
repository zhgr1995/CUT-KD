CUT-KD: Coverage-Aware Utility-Weighted Two-Stage Knowledge Distillation

长尾识别中的覆盖感知、效用加权双阶段知识蒸馏框架


🧠 Overview / 概述
CUT-KD 是一个用于长尾分类（long-tailed recognition）的知识蒸馏框架，结合了：
覆盖感知（Coverage-Aware） 的多教师协同；
效用加权（Utility-Weighted） 的知识融合；
熵触发（Entropy-Triggered） 的学生适配分支。
其目标是在类别不平衡的情况下提升尾部类（rare classes）的识别性能，同时保持计算预算不变。
论文链接（推荐阅读）：https://github.com/zhgr1995/CUT-KD


⚙️ Features / 特性
模块	功能描述	英文简述
多教师路由 (Multi-Teacher Routing)	根据类别分段（头/中/尾）动态分配教师网络	Assigns different teachers to sample segments
效用加权融合 (Utility-Weighted Fusion)	用几何加权平均融合教师输出	Geometric mean fusion weighted by teacher utility
一次剪枝 (One-shot Pruning)	移除低效教师网络以减少冲突	Removes weak teachers by competence ratio
熵触发适配器 (Entropy-Triggered Adapter)	不确定样本激活轻量学生分支	Activates adapter branch for uncertain samples
双阶段蒸馏 (Two-Stage Distillation)	早期全局蒸馏 + 后期覆盖蒸馏	Global KD early + coverage-aware KD later


📦 Dependencies / 环境依赖
Python ≥ 3.8
PyTorch ≥ 2.0
torchvision
numpy
scikit-learn
tensorboardX (可选，用于日志)
安装命令：
pip install torch torchvision numpy scikit-learn tensorboardX tqdm


🚀 Usage / 使用方法
1️⃣ 示例命令（CIFAR-10-LT）
python cut-kd.py \
  --dataset cifar10 \
  --datapath D:\CUT-KD\CIFAR\data \
  --lt_dir D:\CUT-KD\CIFAR\data\cifar-10-LT-10 \
  --out_dir D:\CUT-KD\results \
  --epochs 30 \
  --batch_size 256 \
  --lr 0.1 \
  --eta 0.4 \
  --lambda_max 0.8 \
  --rho 0.25 \
  --tau_pct 80
2️⃣ 参数说明 / Key Arguments
参数	含义	默认值
--dataset	数据集名称（cifar10 / cifar100 / imagenet_lt 等）	必填
--epochs	训练轮数	30
--batch_size	批大小	256
--lr	学习率	0.1
--eta	阶段切换点 (0–1)	0.4
--lambda_max	KD 最大权重	0.7
--rho	适配器混合系数	0.2
--tau_pct	熵触发百分位阈值	80
--out_dir	输出目录	./results



📬 Contact / 联系方式
Institute of Big Data, Southwestern University of Finance and Economics
Email: 1221201z5007@smail.swufe.edu.cn
GitHub: https://github.com/zhgr1995/CUT-KD
