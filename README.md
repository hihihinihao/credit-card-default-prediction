# Credit Card Default Prediction

A comparative machine learning study on the **UCI Credit Card Default** dataset, featuring **DNN optimization** with Weighted Binary Cross-Entropy, Focal Loss, and ensemble methods to address class imbalance.

## Project Structure

```
├── code/
│   ├── dnn_optimized.py          # V1: DNN optimization (4 variants)
│   ├── dnn_optimized_v2.py       # V2: Hyperparameter search + 5-seed ensemble
│   ├── Modeling.ipynb            # Baseline models (LR, RF, XGBoost, SVM, KNN, GBDT)
│   └── EDA/                      # Exploratory Data Analysis
│       ├── ML EDA.ipynb
│       └── *.png                 # EDA figures
├── outputs/
│   ├── results/                  # CSV results for all models
│   └── modeling_figures/         # ROC curves, confusion matrices, etc.
├── paper.tex                     # LaTeX paper (IEEEtran format, 6 pages)
└── .gitignore
```

## Key Results

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---------|----------|-----------|--------|-----|---------|--------|
| Baseline DNN | 0.819 | 0.661 | 0.373 | 0.477 | 0.774 | 0.559 |
| V1: Weighted BCE | 0.805 | 0.565 | 0.509 | **0.536** | **0.777** | 0.562 |
| V2: Best Single | 0.797 | 0.542 | 0.520 | 0.531 | 0.772 | 0.550 |
| V2: Ensemble (5 seeds) | 0.787 | 0.516 | **0.553** | 0.534 | 0.775 | **0.561** |

## Methods

- **Baseline models**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM, KNN
- **DNN**: 4-layer MLP with batch normalization, dropout, ReLU
- **Loss functions**: Standard BCE → Weighted BCE → Focal Loss
- **Hyperparameter search**: 24 configurations over loss type, learning rate, dropout
- **Ensemble**: Average of 5 seeds to reduce variance

## Dependencies

- Python 3.8+
- PyTorch, scikit-learn, pandas, numpy, matplotlib, seaborn

## Usage

```bash
# Run V2 hyperparameter search + ensemble
python code/dnn_optimized_v2.py

# Run V1 variant comparison
python code/dnn_optimized.py
```
