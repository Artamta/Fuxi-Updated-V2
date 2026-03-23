# Fuxi Weather Model (V2)

A modular and scalable implementation of the Fuxi weather forecasting model, designed for efficient pretraining and finetuning on large-scale atmospheric datasets. This repository provides a clean, cluster-ready pipeline for experimentation and reproducible research.

---
> 🔗 **Previous Work:**
> This repository is a refactored and improved version of an earlier implementation.
> 👉 https://github.com/Artamta/Fuxi-Weather-Prediction
- --

##  Features

* Modular architecture (models, training, evaluation separated)
* Pretraining and finetuning pipelines
* SLURM-based cluster support
* Config-driven experiments
* Clean and extensible codebase

---

## 📂 Project Structure

```
fuxi-v2/
│── src/            # core implementation
│── configs/        # experiment configurations
│── scripts/        # cluster run scripts
│── notebooks/      # experiments & visualization
│── data/           # datasets (ignored)
│── logs/           # training logs (ignored)
│── results/        # outputs (ignored)
```

---

##  Installation

```bash
git clone https://github.com/Artamta/Fuxi-Updated-V2.git
cd Fuxi-Updated-V2
pip install -r requirements.txt
```

---

## ▶ Usage

### Pretraining

```bash
bash scripts/pretrain.sh
```

### Finetuning

```bash
bash scripts/finetune.sh
```

---

##  Method Overview

The model leverages transformer-based architectures (including Swin-style components) to capture spatiotemporal dependencies in weather data.

The pipeline consists of:

* Pretraining on large-scale datasets
* Finetuning for downstream forecasting tasks
* Evaluation using custom metrics and analysis tools

---

##  Results

Results, logs, and checkpoints are stored locally and excluded from version control for efficiency.

---

## 📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.

---

## 🤝 Acknowledgements

This work was carried out under the guidance of Dr. Bedartha Goswami at the Machine Learning for Climate Lab, IISER Pune. The authors acknowledge the use of IISER Pune HPC resources for computational support.


---

## 📌 Note

This repository is a refactored and production-ready version of an earlier experimental codebase developed over several months.
