import scipy.io as sio
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import interpolate_data


class BatteryDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)  # (N, Seq, Dim)
        if labels is not None:
            self.labels = torch.FloatTensor(labels).view(-1, 1)
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


def load_battery_data(dataset_name, root_path, battery_names, seq_len=256, feature_type='V-I-T'):
    """
    Unified loader for CACLE and MIT.
    dataset_name: 'cacle' or 'mit'
    feature_type: 'V-I-T' (Voltage, Current, Time) or 'V-IC' (Voltage, Incremental Capacity)
    """
    X_list = []
    y_list = []

    print(f"Loading {dataset_name} batteries: {battery_names}...")

    if dataset_name.lower() == 'cacle':
        # CACLE processing logic
        potential_files = [
            os.path.join(root_path, "CS2_1C_Charge_Data_Relaxed.mat"),
            os.path.join(root_path, "CS2_0.5C_Charge_Data_Relaxed.mat")
        ]

        for bat in battery_names:
            # If in (bat, idx) format, CACLE currently only extracts the 'bat' name (CACLE is typically indexed by name)
            if isinstance(bat, (tuple, list)):
                bat = bat[0]

            found = False
            for fpath in potential_files:
                if not os.path.exists(fpath):
                    # print(f"DEBUG: File not found: {fpath}")
                    continue
                try:
                    mat = sio.loadmat(fpath)
                   # Check key values
                    key = bat
                    if key not in mat and key.replace('-', '_') in mat:
                        key = key.replace('-', '_')

                    if key in mat:
                        bat_data = mat[key]
                        n_cycles = bat_data.shape[1]

                        for i in range(n_cycles):
                            try:
                                volt = bat_data[0, i]['Voltage'].astype(float).flatten()
                                curr = bat_data[0, i]['Current'].astype(float).flatten()
                                time = bat_data[0, i]['Time'].astype(float).flatten()
                                cap = float(np.array(bat_data[0, i]['Capacity']).flat[0])

                                if len(volt) < 10: continue
                                # NaN check
                                if np.isnan(volt).any() or np.isnan(curr).any(): continue

                                # Time normalization
                                time = time - time[0]

                                v_res = interpolate_data(volt, seq_len)
                                c_res = interpolate_data(curr, seq_len)
                                t_res = interpolate_data(time, seq_len)

                                feat = np.stack([v_res, c_res, t_res], axis=1)  # (Seq, 3)
                                X_list.append(feat)
                                y_list.append(cap)
                            except Exception as e:
                                print(f"Error parsing cycle {i}: {e}")
                                pass
                        found = True
                        break  # Battery found in this file, proceed to the next
                    else:
                        pass
                except Exception as e:
                    print(f"ERROR: Failed to load {fpath}: {e}")
            if not found:
                print(f"Warning: Battery {bat} not found.")

    elif dataset_name.lower() == 'mit':
        # MIT processing logic
        for item in battery_names:
            
            if isinstance(item, (tuple, list)):
                fname = item[0]
                target_idx = int(item[1])  # Specify which battery to extract
            else:
                fname = item
                target_idx = None  

            fpath = os.path.join(root_path, fname)
            if not os.path.exists(fpath):
                print(f"File not found: {fpath}")
                continue

            try:
                mat = sio.loadmat(fpath)

                if 'battery' in mat:
                    batteries = mat['battery']
                elif 'batch' in mat:
                    batteries = mat['batch']
                else:
                    print(f"Key 'battery' or 'batch' not found in {fname}")
                    continue

                num_bats = batteries.shape[1]

                if target_idx is not None and target_idx >= num_bats:
                    print(f"Warning: Index {target_idx} out of bounds for {fname} (has {num_bats} bats)")
                    continue

                for b_idx in range(num_bats):

                    if target_idx is not None and b_idx != target_idx:
                        continue
                    try:
                        bat_struct = batteries[0, b_idx]

                        if 'cycles' not in bat_struct.dtype.names:
                            continue

                        cycles = bat_struct['cycles']
                        n_cycles = cycles.shape[1]

                        for c_idx in range(n_cycles):
                            try:
                                cycle = cycles[0, c_idx]
                                if 'voltage_V' in cycle.dtype.names:
                                    volt = cycle['voltage_V'].flatten().astype(float)
                                    curr = cycle['current_A'].flatten().astype(float)
                                    time_min = cycle['relative_time_min'].flatten().astype(float)
                                    cap = cycle['capacity'].flatten()[0]
                                else:
                                    continue

                                if len(volt) < 10: continue
                                if np.isnan(volt).any(): continue

                                time_sec = time_min * 60
                                time_sec = time_sec - time_sec[0]

                                v_res = interpolate_data(volt, seq_len)
                                c_res = interpolate_data(curr, seq_len)
                                t_res = interpolate_data(time_sec, seq_len)

                                feat = np.stack([v_res, c_res, t_res], axis=1)
                                X_list.append(feat)
                                y_list.append(cap)
                            except:
                                pass
                    except Exception as e:
                        print(f"Error extracting battery {b_idx} from {fname}: {e}")

            except Exception as e:
                print(f"Error loading {fname}: {e}")

    X = np.array(X_list)
    y = np.array(y_list)

    if len(X) == 0:
        print("Warning: No data loaded!")
        return np.array([]), np.array([])

    if dataset_name.lower() == 'cacle':
        min_val = np.array([2.0, -1.0, 0.0])
        max_val = np.array([5.0, 10.0, 5000.0])
        X = (X - min_val.reshape(1, 1, 3)) / (max_val.reshape(1, 1, 3) - min_val.reshape(1, 1, 3))
    else:
        # MIT
        min_val = np.array([2.0, -10.0, 0.0])
        max_val = np.array([5.0, 10.0, 3600.0])
        X = (X - min_val.reshape(1, 1, 3)) / (max_val.reshape(1, 1, 3) - min_val.reshape(1, 1, 3))

    if len(y) > 0:
        y_max = y.max()
        if y_max > 0:
            y = y / y_max
            # print(f"Normalized Targets by max value: {y_max:.4f}")

    return X, y
