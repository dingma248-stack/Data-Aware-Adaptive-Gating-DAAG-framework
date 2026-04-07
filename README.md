# Data-Aware-Adaptive-Gating-DAAG-framework
Official code for the DAAG framework: An unsupervised domain adaptation approach utilizing Proxy A-distance (PAD) for dynamic network routing in battery SOH estimation.

This repository provides the official PyTorch implementation of the paper:
**"A Data-Aware Adaptive Gating Framework for Robust Cross-Domain SOH Estimation of Electric Vehicle Lithium-Ion Batteries"** *(Submitted to IEEE Transactions on Transportation Electrification)*.

## 📝 Introduction
Accurate State-of-Health (SOH) estimation for lithium-ion batteries is critical but challenging under complex, dynamic operating conditions. Existing Unsupervised Domain Adaptation (UDA) methods rely on fixed transfer strategies, causing negative transfer in low domain-shift scenarios and insufficient alignment in high-shift ones.

To address this, we propose the **Data-Aware Adaptive Gating (DAAG)** framework. 
* **Heterogeneity Measurement**: Dynamically calculates the Proxy $\mathcal{A}$-distance (PAD) to quantify domain shifts.
* **Dual-Layer Gating**: Adaptively routes the feature extraction architecture (light-weight LSTM vs. enhanced multi-scale CNN-Attn) and the domain alignment constraint (Multi-Kernel MMD).
* **Progressive Warm-up**: Coordinates representation learning and cross-domain alignment to improve training stability.

## 📁 Repository Structure

```text
DAAG-Battery-SOH/
├── datasets/
│   └── loaders.py          # Unified data loaders and preprocessing for CALCE and MIT
├── layers/
│   ├── attention.py        # Self-Attention and CausalConv1d modules
│   └── losses.py           # Multi-Kernel MMD, CORAL, and GRL implementations
├── models/
│   ├── backbones.py        # Feature extractors 
│   └── transfer_net.py     # Main DAAG transfer network 
├── utils/
│   └── preprocessing.py    # Time-series interpolation and normalization
├── main.py                 # 🚀 Entry point: PAD calculation, routing, and multi-seed validation
├── train.py                # Core training loop with warm-up and evaluation
└── README.md


⚙️ PrerequisitesPython 3.8+PyTorch 1.10+NumPy, SciPy, Scikit-learn, Matplotlib, tqdmInstall dependencies using:Bashpip install torch numpy scipy scikit-learn matplotlib tqdm
📊 Dataset PreparationThe framework requires the CALCE and MIT battery aging datasets. Place them in the root directory (or specify your custom path). The structure should look like this:PlaintextRoot_Directory/
├── CACLE数据集/
│   ├── CS2_33...
│   └── CS2_35...
└── MIT数据集/
    └── charge/
        ├── min_batch-5.2-5.2-4.8-4.16.mat
        └── min_batch-6-5.6-4.4-3.834.mat

🧠 Hyperparameters & ConfigurationThe core advantage of the DAAG framework is its Data-Aware nature. Most hyper-parameters are dynamically routed based on the measured PAD value in main.py:
PAD Threshold (PAD_THRESHOLD = 0.8): The boundary distinguishing low/high domain shifts.
  If PAD < 0.8 (Low Shift): Utilizes lstm_only architecture, MMD penalty lambda_mmd = 0.0.
  If PAD ≥ 0.8 (High Shift): Utilizes complete (CNN-LSTM-Attn) architecture, MMD penalty lambda_mmd = 0.05.
Multi-Seed Validation: Set to [2026, 42, 1] by default for robust statistical evaluation.
Warm-up Strategy: Configurable in main.py (WARMUP_EPOCHS = 0 by default, can be adjusted to 30/50 for progressive alignment stability tests).

🚀 How to RunTo reproduce the multi-seed experiments across all 12 transfer tasks (covering both CALCE and MIT), simply execute the main script:Bash--python main.py--

During execution, the script will automatically:
1、Load data and measure real-time PAD via PCA and SVM.
2、Trigger the DAAG routing logic (switching architecture and MMD weight).
3、Train the model and evaluate on target domains (RMSE, MAE, $R^2$).
4、Save SOH prediction visualizations (.png files).
