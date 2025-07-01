# FreqDGT: Frequency-Adaptive Dynamic Graph Networks with Transformer for Cross-subject EEG Emotion Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

FreqDGT is a novel deep learning framework for cross-subject EEG emotion recognition that addresses the fundamental challenges of individual variability and dynamic brain connectivity. Our model integrates frequency-adaptive processing, dynamic spatial modeling, and multi-scale temporal disentanglement within a unified architecture.

## Key Features

- **Frequency-Adaptive Processing (FAP)**: Dynamically weights emotion-relevant frequency bands based on neuroscientific evidence
- **Adaptive Dynamic Graph Learning (ADGL)**: Learns input-specific brain connectivity patterns at multiple granularities  
- **Multi-Scale Temporal Disentanglement Network (MTDN)**: Combines hierarchical temporal transformers with adversarial feature disentanglement
- **Cross-subject Robustness**: Explicitly separates emotion-invariant features from subject-specific variations

## Architecture

```
EEG Input → FAP → ADGL → MTDN → Emotion Classification
    ↓         ↓       ↓       ↓
Frequency  Dynamic  Multi-  Feature
Weighting  Graphs   Scale   Disentang.
```

## Requirements

```bash
pip install torch torchvision numpy scipy scikit-learn einops
```

## Quick Start

```python
from FreqDGT import FreqDGT

# Training
python train_freqdgt.py --feature-type rPSD --dataset SEED
```

## Results

| Dataset | Method | Accuracy | F1-Score |
|---------|--------|----------|----------|
| SEED | FreqDGT | **81.1%** | **81.9%** |
| SEED-IV | FreqDGT | **71.9%** | **72.6%** |
| FACED | FreqDGT | **62.3%** | **76.1%** |


## Citation

If you find this work useful, please cite:

```bibtex
@article{freqdgt2024,
  title={FreqDGT: Frequency-Adaptive Dynamic Graph Networks with Transformer for Cross-subject EEG Emotion Recognition},
  author={Li, Yueyang and Gong, Shengyu and Zeng, Weiming and Wang, Nizhuan and Siok, Wai Ting},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and collaborations, please contact: [yueyoung.li@polyu.edu.hk](mailto:yueyoung.li@polyu.edu.hk)
