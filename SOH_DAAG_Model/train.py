import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt

# 导入自定义模块
from datasets.loaders import load_battery_data, BatteryDataset
from models.transfer_net import TransferNet
from layers.losses import MMD_Loss, CORAL_Loss, GRL, DomainDiscriminator

def set_seed(seed=42):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- 新增辅助函数：解析命令行传递的 keys ---
def parse_keys(keys_str):
    """
    解析逗号分隔的字符串。
    如果包含冒号 (e.g., "file.mat:0")，则转换为元组 (filename, index)。
    否则保留为字符串。
    """
    if not keys_str:
        return []

    raw_list = keys_str.split(',')
    parsed_list = []

    for item in raw_list:
        item = item.strip()
        if ':' in item:
            # 格式为 "filename:index"
            parts = item.split(':')
            filename = parts[0]
            try:
                # 尝试将冒号后的部分转为整数索引
                idx = int(parts[1])
                parsed_list.append((filename, idx))
            except ValueError:
                # 如果转换失败，回退为普通字符串
                parsed_list.append(item)
        else:
            # 普通文件名
            parsed_list.append(item)

    return parsed_list


# ----------------------------------------

def run_experiment(args):
    set_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {DEVICE}")

    # 加载数据
    root_cacle = os.path.join(args.root_dir, "CACLE数据集")
    root_mit = os.path.join(args.root_dir, "MIT数据集", "charge")

    # 解析源域和目标域的 keys
    # 无论 dataset 是 cacle 还是 mit，都尝试解析索引格式
    if args.source_keys:
        source_bats = parse_keys(args.source_keys)
    else:
        # 默认值
        if args.dataset == 'cacle':
            source_bats = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
        else:
            source_bats = ["min_batch-3.6-6-5.6-4.755.mat"]

    if args.target_keys:
        target_bats = parse_keys(args.target_keys)
    else:
        # 默认值
        if args.dataset == 'cacle':
            target_bats = ['CS2_33', 'CS2_34']
        else:
            target_bats = ["min_batch-4.8-5.2-5.2-4.16.mat"]

    if args.dataset == 'cacle':
        print(f"CACLE Source: {source_bats}")
        print(f"CACLE Target: {target_bats}")
        sX, sy = load_battery_data('cacle', root_cacle, source_bats, args.seq_len)
        tX, ty = load_battery_data('cacle', root_cacle, target_bats, args.seq_len)


    elif args.dataset == 'mit':
        print(f"MIT Source: {source_bats}")
        print(f"MIT Target: {target_bats}")

        sX, sy = load_battery_data('mit', root_mit, source_bats, args.seq_len)
        tX, ty = load_battery_data('mit', root_mit, target_bats, args.seq_len)

    print(f"Source Shape: {sX.shape}, Target Shape: {tX.shape}")

    if len(sX) == 0 or len(tX) == 0:
        print("Error: Empty dataset.")
        return

    # 数据预处理（归一化）
    # 源域拟合，目标域变换（域适应标准操作）
    scaler_X = MinMaxScaler()

    # 变形以进行缩放: (N, Seq, 3) -> (N*Seq, 3)
    N_s, Seq, Dim = sX.shape
    N_t = tX.shape[0]

    sX_flat = sX.reshape(-1, Dim)
    tX_flat = tX.reshape(-1, Dim)

    sX_scaled = scaler_X.fit_transform(sX_flat).reshape(sX.shape)
    tX_scaled = scaler_X.transform(tX_flat).reshape(tX.shape)

    # 标签归一化
    scaler_y = MinMaxScaler()
    sy_scaled = scaler_y.fit_transform(sy.reshape(-1, 1))
    ty_scaled = scaler_y.transform(ty.reshape(-1, 1))  # 仅用于验证的真实值

    # 数据加载器
    batch_size = args.batch_size
    # 丢弃最后的不完整批次，并打乱数据

    source_loader = DataLoader(BatteryDataset(sX_scaled, sy_scaled), batch_size=batch_size, shuffle=True,
                               drop_last=True)
    target_train_loader = DataLoader(BatteryDataset(tX_scaled, ty_scaled), batch_size=batch_size, shuffle=True,
                                     drop_last=True)
    target_test_loader = DataLoader(BatteryDataset(tX_scaled, ty_scaled), batch_size=batch_size, shuffle=False)


    model = TransferNet(ablation_mode=args.ablation, input_dim=Dim, hidden_dim=64).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器 - 动态调整: 每 1/3 总 Epochs 衰减一次
    step_size = max(30, int(args.epochs / 3))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    criterion_mse = nn.MSELoss()
    criterion_mmd = MMD_Loss()

    # 训练循环
    print("Starting Training...")

    best_loss = float('inf')  # 用于记录最小的训练 Loss
    rmse_at_best_loss = float('inf')  # 记录在该 Loss 下的目标域 RMSE
    mae_at_best_loss = float('inf')
    r2_at_best_loss = -float('inf')


    history = {'loss': [], 'mse': [], 'mmd': [], 'val_rmse': []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        mse_accum = 0
        mmd_accum = 0
        batches = 0

        iter_source = iter(source_loader)
        iter_target = iter(target_train_loader)
        num_batches = min(len(source_loader), len(target_train_loader))

        for _ in range(num_batches):
            try:
                data_s, label_s = next(iter_source)
                data_t, _ = next(iter_target)
            except StopIteration:
                break

            data_s, label_s = data_s.to(DEVICE), label_s.to(DEVICE)
            data_t = data_t.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播（源域）
            feat_s, pred_s = model(data_s)
            loss_task = criterion_mse(pred_s, label_s)

            # 前向传播
            feat_t, _ = model(data_t)

            # MMD 损失
            loss_mmd = criterion_mmd(feat_s, feat_t)

            current_lambda = args.lambda_mmd
            if args.warmup_epochs > 0:
                if epoch < args.warmup_epochs:
                    # 线性增长：从 0 增长到 lambda_mmd
                    current_lambda = args.lambda_mmd * (epoch / args.warmup_epochs)
                else:
                    current_lambda = args.lambda_mmd

            # 总损失 (使用动态权重的 current_lambda)
            loss = loss_task + current_lambda * loss_mmd

            # # 总损失
            # loss = loss_task + args.lambda_mmd * loss_mmd

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
            optimizer.step()

            total_loss += loss.item()
            mse_accum += loss_task.item()
            mmd_accum += loss_mmd.item()
            batches += 1

        scheduler.step()

        # 验证
        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for data, label in target_test_loader:
                data = data.to(DEVICE)
                _, pred = model(data)
                preds.append(pred.cpu().numpy())
                truths.append(label.numpy())

        preds = np.concatenate(preds)
        truths = np.concatenate(truths)

        # 反归一化
        preds_real = scaler_y.inverse_transform(preds)
        truths_real = scaler_y.inverse_transform(truths)

        mse_val = mean_squared_error(truths_real, preds_real)  # 计算 MSE
        rmse = np.sqrt(mse_val)  # 计算 RMSE
        r2_val = r2_score(truths_real, preds_real)  # 计算 R2
        mae_val = mean_absolute_error(truths_real, preds_real)  # 计算 MAE
        current_epoch_loss = total_loss / batches

        # 日志记录 (保持不变)
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | Loss: {current_epoch_loss:.4f} (MSE: {mse_accum / batches:.4f} MMD: {mmd_accum / batches:.4f}) | Val RMSE: {rmse:.4f} | Val MAE: {mae_val:.4f} | Val R2: {r2_val:.4f}")
        if current_epoch_loss < best_loss:
            best_loss = current_epoch_loss

            # 顺便记录一下这个“最佳Loss模型”对应的 RMSE 是多少，用于最终汇报
            rmse_at_best_loss = rmse
            mae_at_best_loss = mae_val
            r2_at_best_loss = r2_val

            if not os.path.exists("models"):
                os.makedirs("models")

            # 保存模型
            torch.save(model.state_dict(), f"models/best_model_{args.dataset}.pth")

    # print("=" * 50)
    # print(f"Final Best RMSE: {best_rmse:.5f}, Final Best MSE: {best_mse:.5f}, Final Best R2: {best_r2:.5f}")
    # print("=" * 50)

    print("=" * 50)
    print(f"Selection Criterion: Minimum Training Loss")
    print(f"Best Training Loss: {best_loss:.5f}")
    print(f"Target RMSE at Best Loss: {rmse_at_best_loss:.5f}")
    print(f"Target MAE at Best Loss: {mae_at_best_loss:.5f}")
    print(f"Target R2 at Best Loss: {r2_at_best_loss:.5f}")
    print("=" * 50)

    # 重新加载最佳模型并生成预测数据
    print("Reloading best model for plotting...")
    best_model_path = f"models/best_model_{args.dataset}.pth"
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # 使用最佳模型重新进行推理
    preds = []
    truths = []
    with torch.no_grad():
        for data, label in target_test_loader:
            data = data.to(DEVICE)
            _, pred = model(data)
            preds.append(pred.cpu().numpy())
            truths.append(label.numpy())

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    # 反归一化
    preds_real = scaler_y.inverse_transform(preds)
    truths_real = scaler_y.inverse_transform(truths)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(truths_real, label='True Capacity (Ah)', color='black')
    plt.plot(preds_real, label='Predicted Capacity (Ah)', color='red', alpha=0.7)
    plt.title(f"SOH Prediction ({args.dataset.upper()})\nRMSE: {rmse_at_best_loss:.4f}")
    plt.legend()
    # 保存图片
    if args.exp_name:
        save_name = f"result_{args.exp_name}.png"
        plt.title(f"SOH Prediction ({args.exp_name})\nRMSE: {rmse_at_best_loss:.4f}")
    else:
        s_tag = "custom" if args.source_keys else "default"
        save_name = f"result_{args.dataset}_{s_tag}_{np.random.randint(100)}.png"
        plt.title(f"SOH Prediction ({args.dataset.upper()})\nRMSE: {rmse_at_best_loss:.4f}")

    plt.legend()
    plt.savefig(save_name)
    # 同时覆盖主结果文件以便查看 (可选)
    # plt.savefig(f"result_{args.dataset}.png")
    print(f"Plot saved to {save_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # G1 最佳设置
    parser.add_argument('--dataset', type=str, default='cacle', choices=['cacle', 'mit'])
    # 添加实验名称参数
    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment for plot filename')
    # 默认路径设为当前目录
    parser.add_argument('--root_dir', type=str, default=".")
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs for MMD warm-up')####
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lambda_mmd', type=float, default=0.03)

    parser.add_argument('--ablation', type=str, default='complete',
                        choices=['complete', 'no_cnn', 'no_attn', 'lstm_only'],
                        help='Ablation study mode')
    parser.add_argument('--seed', type=int, default=1)
    # 自定义分组的新参数
    parser.add_argument('--source_keys', type=str, default='',
                        help='Comma separated battery names or filenames (e.g. file1,file2:0)')
    parser.add_argument('--target_keys', type=str, default='', help='Comma separated battery names or filenames')

    args = parser.parse_args()

    run_experiment(args)