# **DAGFed: A Dual-Perturbation and Adaptive Graph Framework for Secure and Robust Distributed Learning**

This repository contains the official implementation of **DAG_Fed**, a federated learning framework integrating **Dynamic Adaptive Gradient Clipping (DAGC)** and **Hierarchical Differential Privacy (HDP)** to improve both **privacy protection** and **model utility** under heterogeneous (non-IID) data distributions.

---

## **📌 Features**

* **Dynamic Client Selection & Adaptive Gradient Clipping** — Mitigates statistical heterogeneity by adaptively tuning gradient norms per client.
* **Hierarchical Differential Privacy (HDP)** — Multi-level noise injection for balanced **privacy–utility trade-off**.
* **Modular Aggregator** — Supports **FedAvg**, **SCAFFOLD**, and custom aggregation strategies.
* **Flexible Data Partitioning** — IID and non-IID splits via Dirichlet sampling.
* **Reproducible Experiments** — Unified config file and logging system.

---

## **📂 Project Structure**

```
DAGC_HDP_Federated_Learning/
│
├── config.py                 # Global configuration: hyperparameters, DP budget, scenarios
├── main.py                   # Main entry: training orchestration, global loop
│
├── clients/
│   ├── client.py             # Client base class: data loading, local training, DAGC
│   ├── dp_utils.py           # DP utilities: gradient clipping, noise injection
│   ├── noise_scheduler.py    # Dynamic noise scheduling
│
├── server/
│   ├── aggregator.py         # Aggregators (FedAvg, SCAFFOLD, etc.)
│   ├── privacy_accountant.py # DP budget tracking, RDP accounting
│
├── models/
│   ├── model_cnn.py          # CNN model for vision datasets (CIFAR-10, MNIST)
│
├── data/
│   ├── data_loader.py        # Dataset loading & non-IID partitioning
│   ├── datasets/             # Dataset storage
│
├── experiments/
│   ├── logger.py             # Logging utilities
│   ├── metrics.py            # Accuracy, F1, MSE calculation
│   ├── attack_test.py        # Privacy attack evaluation scripts
│
├── utils/
│   ├── helper.py             # General utilities (seed, save/load, etc.)
│   ├── plots.py              # Visualization of results
│
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## **⚙️ Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/yourusername/DAGC_HDP_Federated_Learning.git
cd DAGC_HDP_Federated_Learning
```

### **2. Create a Python environment**

```bash
conda create -n dagc_hdp python=3.9
conda activate dagc_hdp
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **🚀 Usage**

Run a CIFAR-10 experiment with 10 clients, 100 rounds, and a privacy budget of ε = 8.0:

```bash
python main.py --dataset cifar10 --rounds 100 --clients 10 --epsilon 8.0
```

**Key Arguments**:

* `--dataset` : `cifar10`, `cifar100`, `mnist`, `fashionmnist`
* `--rounds` : Number of global communication rounds
* `--clients` : Number of participating clients
* `--epsilon` : Privacy budget for DP
* Additional parameters can be set in `config.py`

---

