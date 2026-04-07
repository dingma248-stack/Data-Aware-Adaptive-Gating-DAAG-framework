import subprocess
import re
import sys
import os
import time
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.utils import resample

try:
    from tqdm import tqdm
except ImportError:
    print("It is recommended to install tqdm for better progress visualization: pip install tqdm")
    tqdm = None

try:
    from datasets.loaders import load_battery_data
except ImportError:
    try:
        from loaders import load_battery_data
    except ImportError:
        print("Warning: load_battery_data function not found, please ensure it is in the Python path")


C_33, C_34, C_35, C_36, C_37, C_38 = "CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"
MIT_FILE_A = "min_batch-5.2-5.2-4.8-4.16.mat"
MIT_FILE_B = "min_batch-6-5.6-4.4-3.834.mat"

EXPERIMENTS_LIST = []

# 1. CACLE 
cacle_source_train = [C_35, C_36, C_37, C_38]
cacle_targets = [C_33, C_34]
for tgt in cacle_targets:
    EXPERIMENTS_LIST.append({
        "mode_name": f"CACLE_ALL_to_{tgt}",
        "dataset": "cacle",
        "exp_name": f"CACLE_Exp_ALL_to_{tgt}",
        "source": cacle_source_train,
        "target": [tgt]
    })

cacle_source_rev = [C_33, C_34]
cacle_targets_rev = [C_35, C_36]
for tgt in cacle_targets_rev:
    EXPERIMENTS_LIST.append({
        "mode_name": f"CACLE_Rev_ALL_to_{tgt}",
        "dataset": "cacle",
        "exp_name": f"CACLE_Exp_Rev_ALL_to_{tgt}",
        "source": cacle_source_rev,
        "target": [tgt]
    })

# 2. MIT 
mit_configs = [
    ((MIT_FILE_A, 0), (MIT_FILE_B, 0)), ((MIT_FILE_A, 0), (MIT_FILE_B, 1)),
    ((MIT_FILE_A, 1), (MIT_FILE_B, 0)), ((MIT_FILE_A, 1), (MIT_FILE_B, 1)),
    ((MIT_FILE_B, 0), (MIT_FILE_A, 0)), ((MIT_FILE_B, 1), (MIT_FILE_A, 1)),
    ((MIT_FILE_B, 0), (MIT_FILE_A, 1)), ((MIT_FILE_B, 1), (MIT_FILE_A, 0))
]

for i, (src, tgt) in enumerate(mit_configs):
    s_name = "FileA" if src[0] == MIT_FILE_A else "FileB"
    t_name = "FileA" if tgt[0] == MIT_FILE_A else "FileB"
    EXPERIMENTS_LIST.append({
        "mode_name": f"MIT_{s_name}idx{src[1]}_to_{t_name}idx{tgt[1]}",
        "dataset": "mit",
        "exp_name": f"MIT_Exp_{i + 1}_{s_name}_{src[1]}_to_{t_name}_{tgt[1]}",
        "source": [src],
        "target": [tgt]
    })


WARMUP_EPOCHS = 0
SEEDS = [2026, 42, 1]
PAD_THRESHOLD = 0.8


def calculate_proxy_a_distance(source_data, target_data):
    X_s = source_data.reshape(source_data.shape[0], -1)
    X_t = target_data.reshape(target_data.shape[0], -1)

    min_len = min(len(X_s), len(X_t))
    if min_len == 0:
        return 0.0

    X_s = resample(X_s, n_samples=min_len, random_state=42)
    X_t = resample(X_t, n_samples=min_len, random_state=42)

    y_s = np.zeros(X_s.shape[0])
    y_t = np.ones(X_t.shape[0])
    X = np.vstack((X_s, X_t))
    y = np.hstack((y_s, y_t))

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)

    clf = LinearSVC(C=0.05, random_state=42, max_iter=10000, dual=False)
    scores = cross_val_score(clf, X_pca, y, cv=5)
    generalization_error = 1.0 - np.mean(scores)

    pad = 2.0 * (1.0 - 2.0 * min(generalization_error, 0.5))
    return max(0.0, pad)


def get_adaptive_config_by_pad(pad_value):
    if pad_value < PAD_THRESHOLD:
        return {
            "ablation": "lstm_only",
            "lambda_mmd": 0.0,
            "epochs": 300,
            "lr": 0.001,
            "shift_level": "LOW"
        }
    else:
        return {
            "ablation": "complete",
            "lambda_mmd": 0.05,
            "epochs": 300,
            "lr": 0.001,
            "shift_level": "HIGH"
        }

def run_experiment_task(task_config, current_seed, config, pad_value, project_root):
    dataset_type = task_config["dataset"]
    unique_exp_name = f"{task_config['exp_name']}_s{current_seed}"

    sys.stdout.write("\n")
    sys.stdout.flush()
    print(f"{'-' * 70}")
    print(f"[DAAG Routing] Measured PAD: {pad_value:.4f} -> Shift Level: {config['shift_level']}")
    print(f"[Configuration] Arch: {config['ablation']:<10} | MMD Penalty: {config['lambda_mmd']:.2f}")
    print(f" >>> Running Seed: {current_seed}")
    print(f"{'-' * 70}")

    python_path = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train.py")

    def format_keys(keys_list):
        return ",".join(
            [f"{item[0]}:{item[1]}" if isinstance(item, (list, tuple)) else str(item) for item in keys_list])
    cmd_args = [
        "--dataset", dataset_type,
        "--root_dir", project_root,
        "--exp_name", unique_exp_name,
        "--lr", str(config["lr"]),
        "--epochs", str(config["epochs"]),
        "--lambda_mmd", str(config["lambda_mmd"]),
        "--seed", str(current_seed),
        "--source_keys", format_keys(task_config["source"]),
        "--target_keys", format_keys(task_config["target"]),
        "--warmup_epochs", str(WARMUP_EPOCHS),
        "--ablation", config["ablation"]
    ]

    full_cmd = [python_path, train_script] + cmd_args

    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    final_rmse, final_mae, final_r2 = None, None, None

    if tqdm:
        pbar = tqdm(total=config['epochs'], desc=f"Seed {current_seed}", unit="ep", dynamic_ncols=True, position=0,
                    leave=True, file=sys.stdout)
    else:
        pbar = None

    last_epoch_num = 0

    try:
        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()

            if "Epoch" in clean_line and "/" in clean_line:
                loss_match = re.search(r"Loss:\s*([\d.]+)", clean_line)
                rmse_match = re.search(r"Val RMSE:\s*([\d.]+)", clean_line)
                mae_match = re.search(r"Val MAE:\s*([\d.]+)", clean_line)
                r2_match = re.search(r"Val R2:\s*([\d.\-]+)", clean_line)
                epoch_match = re.search(r"Epoch\s+(\d+)/", clean_line)

                if pbar and epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    if current_epoch > last_epoch_num:
                        pbar.update(current_epoch - last_epoch_num)
                        last_epoch_num = current_epoch

                    pbar.set_postfix({
                        "L": loss_match.group(1) if loss_match else "?",
                        "RMSE": rmse_match.group(1) if rmse_match else "?",
                        "R2": r2_match.group(1) if r2_match else "?"
                    })

            elif "Target RMSE at Best Loss" in clean_line:
                if pbar: pbar.write(f"[Result] {clean_line}")
                match = re.search(r"Target RMSE at Best Loss: ([0-9.nan]+)", clean_line)
                if match: final_rmse = float('nan') if match.group(1) == 'nan' else float(match.group(1))

            elif "Target MAE at Best Loss" in clean_line:
                match = re.search(r"Target MAE at Best Loss: ([0-9.nan\-]+)", clean_line)
                if match: final_mae = float('nan') if match.group(1) == 'nan' else float(match.group(1))

            elif "Target R2 at Best Loss" in clean_line:
                match = re.search(r"Target R2 at Best Loss: ([0-9.nan\-]+)", clean_line)
                if match: final_r2 = float('nan') if match.group(1) == 'nan' else float(match.group(1))

            elif "Best Training Loss" in clean_line:
                if pbar: pbar.write(f"[Result] {clean_line}")

    except Exception as e:
        print(f"\n[Error] Failed to read process output: {e}")
    finally:
        if pbar:
            pbar.close()
            sys.stdout.flush()
            time.sleep(0.5)

    process.wait()
    return final_rmse, final_mae, final_r2

def run_multi_seed_experiment():
    print(f"\n=======================================================================")
    print(f" Starting DAAG Multi-Seed Validation ({len(SEEDS)} seeds) with {len(EXPERIMENTS_LIST)} tasks...")
    print(f" Gating Logic: True PAD measurement (Threshold = {PAD_THRESHOLD})")
    print(f"=======================================================================")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, "CACLE")):
        project_root = script_dir
    elif os.path.exists(os.path.join(os.path.dirname(script_dir), "CACLE")):
        project_root = os.path.dirname(script_dir)
    else:
        project_root = script_dir

    root_cacle = os.path.join(project_root, "CACLE")
    root_mit = os.path.join(project_root, "MIT", "charge")

    final_results = {}

    for task in EXPERIMENTS_LIST:
        task_name = task["mode_name"]
        dataset_type = task["dataset"]

        print(f"\n\n{'=' * 80}")
        print(f"PROCESSING TASK: {task_name}")

        print(f"[Phase 1] Loading data to calculate Proxy A-distance...")
        root_path = root_cacle if dataset_type == 'cacle' else root_mit
        try:
            X_s, _ = load_battery_data(dataset_type, root_path, task['source'], seq_len=256)
            X_t, _ = load_battery_data(dataset_type, root_path, task['target'], seq_len=256)

            if len(X_s) == 0 or len(X_t) == 0:
                print(f"Warning: Data loading failed for task {task_name}, skipping computation.")
                continue

            pad_val = calculate_proxy_a_distance(X_s, X_t)
        except Exception as e:
            print(f"Error occurred during PAD calculation: {e}. Falling back to the default low domain bias configuration.")
            pad_val = 0.0

        task_config_dict = get_adaptive_config_by_pad(pad_val)

        print(f"[Phase 2] Starting model training over {len(SEEDS)} seeds...")
        print(f"{'=' * 80}")
        seed_rmses, seed_maes, seed_r2s = [], [], []

        for seed in SEEDS:
            rmse, mae, r2 = run_experiment_task(task, seed, task_config_dict, pad_val, project_root)
            if rmse is not None and not np.isnan(rmse):
                seed_rmses.append(rmse)
                seed_maes.append(mae)
                seed_r2s.append(r2)
            else:
                print(f"Warning: Task {task_name} with Seed {seed} Failed or NaN.")
            time.sleep(1)

        final_results[task_name] = {
            "rmse": seed_rmses,
            "mae": seed_maes,
            "r2": seed_r2s,
            "pad": pad_val
        }

    print(f"\n\n{'=' * 125}")
    print(
        f"{'Experiment Task':<35} | {'PAD':<6} | {'Mean RMSE ± Std':<22} | {'Mean MAE ± Std':<22} | {'Mean R2 ± Std':<22}")
    print(f"{'-' * 125}")

    for name, metrics in final_results.items():
        rmses, maes, r2s, pad_val = metrics["rmse"], metrics["mae"], metrics["r2"], metrics["pad"]

        if len(rmses) > 0:
            str_rmse = f"{np.mean(rmses):.5f} ± {np.std(rmses):.5f}"
            str_mae = f"{np.mean(maes):.5f} ± {np.std(maes):.5f}"
            str_r2 = f"{np.mean(r2s):.5f} ± {np.std(r2s):.5f}"
            pad_str = f"{pad_val:.2f}"

            print(f"{name:<35} | {pad_str:<6} | {str_rmse:<22} | {str_mae:<22} | {str_r2:<22}")
        else:
            print(f"{name:<35} | Failed (All seeds)")
    print(f"{'=' * 125}\n")


if __name__ == "__main__":
    run_multi_seed_experiment()
