# Efficient Solutions for Mitigating Initialization Bias in Unsupervised Self-Adaptive Auditory Attention Decoding

This repository contains the implementation code for [1] (Accepted for ICASSP 2026).

## Overview

Auditory Attention Decoding (AAD) aims to identify the attended speaker in multi-speaker environments using brain signals such as EEG. This project implements computationally efficient methods to mitigate initialization bias in unsupervised self-adaptive AAD systems based on Canonical Correlation Analysis (CCA).

More specifically, the paper proposes three efficient alternatives to the cross-validated unbiased method:

1. **Two-Encoder Version**: Trains encoders for both attended and unattended speech features, sharing the same EEG decoder
2. **Soft Version**: Replaces hard segment assignments with probabilistic weights based on prediction uncertainty
3. **Sum-Initialized Single-Encoder**: Initializes the model with a composite signal (sum of both speakers' features)

Use the provided `environment.yaml` to create a conda environment with all necessary dependencies.

## Running Experiments

The main experiment script is `speech_iter.py`. You can run experiments using:

```bash
python speech_iter.py --dataset fuglsang2018 --method single --nbtraintrials 24 --shuffle
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (`fuglsang2018`, `Neetha`) | Required |
| `--method` | AAD method to use | Required |
| `--nbtraintrials` | Number of training trials | None (use all) |
| `--predtriallen` | Prediction trial length (seconds) | Dataset-specific |
| `--updatewinlen` | Update window length (seconds) | Dataset-specific |
| `--hparadata` | Hankel matrix params for EEG `[L, offset]` | Dataset-specific |
| `--hparafeats` | Hankel matrix params for audio `[L, offset]` | Dataset-specific |
| `--evalpara` | Evaluation params `[range, nb_comp]` | `[3, 2]` |
| `--shuffle` | Shuffle trials | `False` |
| `--seeds` | Random seeds for multiple runs | `[16, 32, 64, 128]` |

### Available Methods

| Method | Description |
|--------|-------------|
| `single` | Single-encoder baseline (biased) |
| `single_warminit` | Sum-initialized single-encoder (proposed) |
| `two` | Two-encoder version (proposed) |
| `bpsk_local` | Soft version with local probability estimation (proposed) |
| `single_unbiased` | Cross-validated unbiased method (reference) |

### Batch Experiments

Use `runexperiments.py` to run multiple experiments in parallel:

```bash
python runexperiments.py
```

This script runs experiments across different training set sizes and methods, storing results in the `tables/` folder. The results can be visualized by running `res_visual.ipynb`.

## Supported Datasets

1. **Neetha [2]**: 64-channel EEG, 20 Hz, 60s trials. Results are included in the paper. Publicly available at https://zenodo.org/records/4004271.
2. **fuglsang2018 [3]**: 64-channel EEG, 32 Hz, 50s trials. Results are not included in the paper but can be found in this repository (res_fuglsang2018.pdf). Publicly available at https://zenodo.org/records/1199011.

## Key Classes

- `MCCA_LW` (`algo_suppl.py`): (Multi-view) CCA with Ledoit-Wolf covariance regularization
- `CorrelationAnalysis` (`algo_ccazoo.py`): High-level CCA analysis with cross-validation
- `ITERATIVE` (`utils_iter.py`): Iterative unsupervised AAD implementation
- `STREAM` (`utils_stream.py`): Time-adaptive unsupervised AAD implementation (Not used in this paper)

## Citation

If you use this code, please cite:

```bibtex
@misc{yao2025efficient,
      title={Efficient Solutions for Mitigating Initialization Bias in Unsupervised Self-Adaptive Auditory Attention Decoding}, 
      author={Yuanyuan Yao and Simon Geirnaert and Tinne Tuytelaars and Alexander Bertrand},
      year={2025},
      eprint={2509.14764},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2509.14764}, 
}
```

## References

[1] Yao, Y., Geirnaert, S., Tuytelaars, T., & Bertrand, A. (2025). Efficient Solutions for Mitigating Initialization Bias in Unsupervised Self-Adaptive Auditory Attention Decoding. arXiv preprint arXiv:2509.14764.

[2] Biesmans, W., Das, N., Francart, T., & Bertrand, A. (2016). Auditory-inspired speech envelope extraction methods for improved EEG-based auditory attention detection in a cocktail party scenario. IEEE transactions on neural systems and rehabilitation engineering, 25(5), 402-412.

[3] Fuglsang, S. A., Dau, T., & Hjortkjær, J. (2017). Noise-robust cortical tracking of attended speech in real-life environments. NeuroImage, 156, 435-444.
