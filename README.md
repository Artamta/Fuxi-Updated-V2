# FuXi Weather Model (V2)

FuXi-inspired global weather forecasting with a Swin + U-Net pipeline, autoregressive diagnostics, and cluster-ready training workflows.

<p align="center">
	<img src="prof/report/gifs_daybyday_init0_15d/t2m_rollout_15d_init0_20210305T1800.gif" alt="T2M rollout (15-day)" width="88%" />
</p>

<p align="center"><i>T2M rollout (15-day).</i></p>

<p align="center">
	<img src="prof/report/gifs_daybyday_init0_15d/rich_rollout_15d_init0_20210305T1800.gif" alt="Rich-variable rollout (15-day)" width="88%" />
</p>

<p align="center"><i>Rich-variable rollout (15-day).</i></p>

> For more information, refer to [report/report.pdf](report/report.pdf) and [report/report.tex](report/report.tex).

---

## Previous Work

This repository is a refactored and improved version of:
https://github.com/Artamta/Fuxi-Weather-Prediction

## Highlights

- Modular architecture with clear separation of model, training, and evaluation
- Pretraining and finetuning pipelines
- SLURM-ready scripts for HPC execution
- Config-driven experiments for reproducibility
- Diagnostics for RMSE, ACC, and long-horizon rollout behavior

## Method Overview

The model follows a FuXi-inspired design:
- 3D cube embedding for spatiotemporal tokenization
- Swin-style transformer blocks for global context
- U-Net-style decoder for spatial reconstruction
- Autoregressive rollout analysis for lead-time skill trends

## Project Structure

```text
fuxi_new/
|-- src/                  core implementation
|-- configs/              experiment configurations
|-- scripts/              local and cluster run scripts
|-- report/               report source and compiled document
|-- prof/report/          gifs and diagnostics used in analysis
|-- notebooks/            experiments and visualization
|-- data/                 datasets (not tracked)
|-- logs/                 training logs (not tracked)
|-- results/              outputs (not tracked)
`-- results_new/          refreshed evaluation outputs
```

## Installation

```bash
git clone https://github.com/Artamta/Fuxi-Updated-V2.git
cd Fuxi-Updated-V2
pip install -r requirements.txt
```

## Usage

### Pretraining

```bash
bash scripts/pretrain.sh
```

### Finetuning

```bash
bash scripts/finetune.sh
```

### Example Evaluation Comparison

```bash
bash scripts/run_eval_compare_single_gpu.sh
```

## Results

Model artifacts, logs, and checkpoints are generated locally and excluded from version control.

## Report

- Full report: [report/report.pdf](report/report.pdf)
- Source: [report/report.tex](report/report.tex)

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

## Acknowledgements

This work was carried out under the guidance of Dr. Bedartha Goswami at the Machine Learning for Climate Lab, IISER Pune. The authors acknowledge IISER Pune HPC resources for computational support.
