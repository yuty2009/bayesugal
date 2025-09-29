# Efficient Sleep Staging with Bayesian Uncertainty-Guided Active Learning

<!-- ## Abstract
Automated sleep staging is essential for large-scale and home-based sleep monitoring, yet current machine learning-based systems remain clinically limited—particularly due to poor generalization across subjects, recording centers, and wearable sleep monitoring devices. As a result, clinicians still rely heavily on costly manual scoring in practice. This creates an urgent need for adaptive, efficient, and reliable systems that minimize labeling effort without compromising accuracy. We propose BayesSleepNet, the first framework to combine Bayesian uncertainty quantification with active learning for adaptive sleep staging. Unlike conventional confidence scores derived from softmax outputs—such as hypnodensity maps—BayesSleepNet employs principled Bayesian modeling by placing distributions over network weights and performing Monte Carlo sampling at inference to explicitly quantify both epistemic and aleatoric uncertainty. These uncertainty estimates are then used to drive a two-stage sample selection strategy that first fine-tunes the model with representative samples and subsequently targets persistently uncertain cases for expert annotation.
Across four public sleep datasets, BayesSleepNet achieves consistent performance improvements—+7.60\% in accuracy, +8.27\% in macro-F1, and +0.104 in Cohen's $\kappa$—using only 20\% labeled data from new subjects. Despite its strong performance, BayesSleepNet maintains a lightweight design with over 10× fewer parameters than state-of-the-art models. These results underscore the clinical promise of uncertainty-aware active learning in developing scalable, accurate, and cost-efficient sleep staging systems. Code is available at https://github.com/yuty2009/bayesugal. -->

## Method Overview
- **Bayesian Neural Networks**: We employ Bayesian convolutional neural networks with MC Dropout to estimate predictive uncertainty.
- **Uncertainty-Guided Active Learning**: At each active learning iteration, samples with the highest model uncertainty (entropy or Bayesian uncertainty) are selected for expert annotation.
- **Fine-Tuning**: The model is fine-tuned on the newly labeled samples, iteratively improving performance with minimal annotation effort.
- **Multi-Model Ensemble**: Multiple models are trained and ensembled to further improve robustness and uncertainty estimation.

## Datasets Supported
- Sleep-EDF (20 and 78 subjects)
- MASS
- Physio2018

## Key Features
- End-to-end pipeline for active learning in sleep staging
- Flexible support for multiple datasets and experimental settings
- Modular code for data loading, model training, active learning, and evaluation
- Comprehensive evaluation: Overall Accuracy, Macro F1, Cohen's Kappa
- Publication-quality visualization scripts

## Directory Structure
```
BayesianUGAL/
├── bayesian_ugal_*.py           # Main model and experiment scripts
├── train_active_learning_*.py   # Active learning training scripts
├── test-*.py                    # Testing scripts
├── tools.py                     # Core utility functions
├── EDFreader/                   # EDF data reading and processing
├── pre/                         # Preprocessing and feature engineering
├── physio2018/                  # Physio2018 related scripts
├── PSG_process/                 # PSG file processing tools
├── requirements.txt             # Dependency list
```

## Installation & Requirements
- Python 3.8+
- PyTorch >= 1.8
- numpy, tqdm, matplotlib, scikit-learn, pyEDFlib, etc.
- (Recommended) Anaconda environment

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Place raw datasets in the specified folders, or use scripts in `EDFreader/` and `pre/` for preprocessing.
2. **Pretraining**: Train the initial model on labeled data:
   ```bash
   python train-pretrained-model.py
   ```
3. **Active Learning**: Run the active learning loop:
   ```bash
   python train_active_learning.py
   ```
4. **Evaluation**: Test and evaluate the model:
   ```bash
   python test-pretrained-model.py
   ```
5. **Visualization**: Generate figures for analysis:
   ```bash
   python sleep_stage_figure.py
   ```

## Main Scripts
- `train-pretrained-model.py`: Pretraining on initial labeled set
- `train_active_learning.py`: Active learning and fine-tuning
- `test-pretrained-model.py`: Model evaluation
- `tools.py`: Core training, testing, and active learning utilities
- `EDFreader/`, `pre/`: Data loading and preprocessing

## Evaluation Metrics
- **Overall Accuracy**
- **Macro F1 (MF1)**
- **Cohen's Kappa**

<!-- ## Citation
If you use this code or method in your research, please cite our paper:
```
@article{YourPaper2025,
  title={Efficient Sleep Staging with Bayesian Uncertainty-Guided Active Learning},
  author={Your Name et al.},
  journal={Journal/Conference Name},
  year={2025}
}
``` -->

## Acknowledgements
- Public datasets: Sleep-EDF, MASS, ISRUC, Physio2018
- Open source tools: PyTorch, scikit-learn, pyEDFlib, etc.

---
For questions or collaboration, please contact the project authors.
