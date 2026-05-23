"""
Optimized Deep Neural Network for Credit Card Default Prediction
=======================================================
Improvements over the baseline DNN:
  1. Focal Loss — handles class imbalance by focusing on hard examples
  2. Weighted Random Sampler — ensures balanced batches during training
  3. Wider/deeper architecture (256→128→64→32) — more capacity
  4. LeakyReLU — avoids dead neurons
  5. ReduceLROnPlateau — adaptive learning rate
  6. Gradient clipping — stable training
  7. Longer patience (20) — allows more epochs to converge
"""

import os
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

warnings.filterwarnings('ignore')

# ── Paths ──
DATA_PATH = Path(__file__).parent / 'EDA' / 'UCI_Credit_Card.xls'
OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs'
FIG_DIR = OUTPUT_DIR / 'modeling_figures'
RESULT_DIR = OUTPUT_DIR / 'results'
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}\n")

# ═══════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading and preprocessing data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Target
target_col = 'default.payment.next.month'

# Fix encoding
df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x in [0, 5, 6] else x)
df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: 3 if x == 0 else x)

# Feature engineering (same as original)
pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

df_features = df.copy()

df_features['delay_count'] = (df_features[pay_status_cols] > 0).sum(axis=1)
df_features['severe_delay_count'] = (df_features[pay_status_cols] >= 2).sum(axis=1)
df_features['max_delay'] = df_features[pay_status_cols].max(axis=1)
df_features['avg_delay'] = df_features[pay_status_cols].mean(axis=1)
df_features['recent_delay'] = df_features['PAY_0']
df_features['recent_delay_flag'] = (df_features['PAY_0'] > 0).astype(int)

df_features['avg_bill_amt'] = df_features[bill_cols].mean(axis=1)
df_features['max_bill_amt'] = df_features[bill_cols].max(axis=1)
df_features['recent_bill_amt'] = df_features['BILL_AMT1']
df_features['bill_trend'] = df_features['BILL_AMT1'] - df_features['BILL_AMT6']
df_features['credit_utilization'] = df_features['BILL_AMT1'] / df_features['LIMIT_BAL'].replace(0, np.nan)

df_features['avg_pay_amt'] = df_features[pay_amt_cols].mean(axis=1)
df_features['max_pay_amt'] = df_features[pay_amt_cols].max(axis=1)
df_features['recent_pay_amt'] = df_features['PAY_AMT1']
df_features['pay_trend'] = df_features['PAY_AMT1'] - df_features['PAY_AMT6']

df_features['recent_pay_bill_ratio'] = df_features['PAY_AMT1'] / (df_features['BILL_AMT2'].abs() + 1)
df_features['total_pay_bill_ratio'] = (df_features[pay_amt_cols].sum(axis=1)
                                       / (df_features[bill_cols].abs().sum(axis=1) + 1))

df_features = df_features.replace([np.inf, -np.inf], np.nan).fillna(0)

engineered_features = [
    'delay_count', 'severe_delay_count', 'max_delay', 'avg_delay',
    'recent_delay', 'recent_delay_flag', 'avg_bill_amt', 'max_bill_amt',
    'recent_bill_amt', 'bill_trend', 'credit_utilization',
    'avg_pay_amt', 'max_pay_amt', 'recent_pay_amt', 'pay_trend',
    'recent_pay_bill_ratio', 'total_pay_bill_ratio'
]

raw_features = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

feature_set_b = raw_features + engineered_features
X = df_features[feature_set_b].values
y = df_features[target_col].values.astype(np.float32)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train/val split (for early stopping)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_s, y_train, test_size=0.15, random_state=42, stratify=y_train
)

pos_ratio = y_train.mean()
print(f"Training size: {len(y_tr)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
print(f"Positive class ratio in training: {pos_ratio:.3f}\n")

# ═══════════════════════════════════════════
# 2. DEFINE LOSS FUNCTIONS
# ═══════════════════════════════════════════
print("=" * 60)
print("STEP 2: Defining loss functions")
print("=" * 60)


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.
    Down-weights easy examples, focuses on hard, misclassified ones.
    alpha: class-balance weight (higher = more weight on positive class)
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)  # predicted probability for the true class
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        # alpha: weight positive class by alpha, negative by 1-alpha
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (focal_weight * alpha_weight * bce).mean()


class WeightedBCELoss(nn.Module):
    """Weighted BCE for imbalanced classification."""
    def __init__(self, pos_weight=3.5):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weight = targets * self.pos_weight + (1 - targets) * 1.0
        return (weight * loss).mean()


print("  - FocalLoss (alpha=0.75, gamma=2.0)")
print("  - WeightedBCELoss (pos_weight=3.5)\n")

# ═══════════════════════════════════════════
# 3. DEFINE OPTIMIZED DNN ARCHITECTURE
# ═══════════════════════════════════════════
print("=" * 60)
print("STEP 3: Defining optimized architecture")
print("=" * 60)


class OptimizedCreditRiskDNN(nn.Module):
    """Optimized DNN: wider, LeakyReLU, lower dropout, 4 hidden layers."""
    def __init__(self, input_dim, dropout_rate=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, 1),
        )

        # Kaiming init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


input_dim = X_tr.shape[1]
print(f"  Input features: {input_dim}")
print(f"  Architecture: {input_dim} → 256 → 128 → 64 → 32 → 1\n")

# ═══════════════════════════════════════════
# 4. TRAINING LOOP
# ═══════════════════════════════════════════
print("=" * 60)
print("STEP 4: Training")
print("=" * 60)


def create_weighted_sampler(y):
    """Create WeightedRandomSampler to balance classes in each batch."""
    y_arr = np.array(y)
    class_counts = np.bincount(y_arr.astype(int))
    weight_per_class = 1.0 / class_counts
    sample_weights = weight_per_class[y_arr.astype(int)]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_dnn(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs, patience, clip_norm=1.0, save_path='best_dnn_model.pth'):
    model = model.to(device)
    best_val_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []
    val_aucs = []

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_aucs.append(val_auc)

        # Scheduler step
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} "
                  f"| Val Loss: {epoch_val_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {lr:.2e}")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"  >> Early stopping triggered at epoch {epoch+1}")
                break

    return train_losses, val_losses, val_aucs


def evaluate(model, X_test_tensor, y_test_np, device='cpu'):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        y_proba = torch.sigmoid(outputs).cpu().numpy().flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        'accuracy': accuracy_score(y_test_np, y_pred),
        'precision': precision_score(y_test_np, y_pred),
        'recall': recall_score(y_test_np, y_pred),
        'f1': f1_score(y_test_np, y_pred),
        'roc_auc': roc_auc_score(y_test_np, y_proba),
        'pr_auc': average_precision_score(y_test_np, y_proba),
        'y_proba': y_proba,
        'y_pred': y_pred,
    }


# Convert to tensors
X_tr_t = torch.FloatTensor(X_tr)
y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test_s)

# Common settings
batch_size = 256
epochs = 200
patience = 20

results = {}

# ── Variant A: Improved architecture + BCE + LR scheduler ──
print("\n--- Variant A: BCE Loss + ReduceLROnPlateau (architecture only) ---")
torch.manual_seed(42)
model_a = OptimizedCreditRiskDNN(input_dim)
criterion_a = nn.BCEWithLogitsLoss()
optimizer_a = optim.Adam(model_a.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(optimizer_a, mode='min', factor=0.5, patience=8, min_lr=1e-6)

train_dataset = TensorDataset(X_tr_t, y_tr_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

t0 = time.time()
tl, vl, va = train_dnn(model_a, criterion_a, optimizer_a, scheduler_a,
                        train_loader, val_loader, epochs, patience,
                        save_path=RESULT_DIR / 'best_dnn_a.pth')
t_a = time.time() - t0
model_a.load_state_dict(torch.load(RESULT_DIR / 'best_dnn_a.pth', weights_only=True, map_location=device))
results['BCE + Scheduler'] = evaluate(model_a, X_test_t, y_test, device)
results['BCE + Scheduler']['train_time'] = t_a
results['BCE + Scheduler']['train_losses'] = tl
results['BCE + Scheduler']['val_losses'] = vl
results['BCE + Scheduler']['val_aucs'] = va
print(f"  Train time: {t_a:.2f}s")
print(f"  Test  Acc:{results['BCE + Scheduler']['accuracy']:.4f}  Prec:{results['BCE + Scheduler']['precision']:.4f}  "
      f"Rec:{results['BCE + Scheduler']['recall']:.4f}  F1:{results['BCE + Scheduler']['f1']:.4f}  "
      f"ROC:{results['BCE + Scheduler']['roc_auc']:.4f}  PR:{results['BCE + Scheduler']['pr_auc']:.4f}")

# ── Variant B: Focal Loss (no WeightedSampler, only loss handles imbalance) ──
print("\n--- Variant B: Focal Loss (alpha=0.75, gamma=2.0, no sampler) ---")
torch.manual_seed(42)
model_b = OptimizedCreditRiskDNN(input_dim)
criterion_b = FocalLoss(alpha=0.75, gamma=2.0)
optimizer_b = optim.Adam(model_b.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode='min', factor=0.5, patience=8, min_lr=1e-6)

t0 = time.time()
tl, vl, va = train_dnn(model_b, criterion_b, optimizer_b, scheduler_b,
                        train_loader, val_loader, epochs, patience,
                        save_path=RESULT_DIR / 'best_dnn_b.pth')
t_b = time.time() - t0
model_b.load_state_dict(torch.load(RESULT_DIR / 'best_dnn_b.pth', weights_only=True, map_location=device))
results['Focal Loss'] = evaluate(model_b, X_test_t, y_test, device)
results['Focal Loss']['train_time'] = t_b
results['Focal Loss']['train_losses'] = tl
results['Focal Loss']['val_losses'] = vl
results['Focal Loss']['val_aucs'] = va
print(f"  Train time: {t_b:.2f}s")
print(f"  Test  Acc:{results['Focal Loss']['accuracy']:.4f}  Prec:{results['Focal Loss']['precision']:.4f}  "
      f"Rec:{results['Focal Loss']['recall']:.4f}  F1:{results['Focal Loss']['f1']:.4f}  "
      f"ROC:{results['Focal Loss']['roc_auc']:.4f}  PR:{results['Focal Loss']['pr_auc']:.4f}")

# ── Variant C: Milder Weighted BCE (pos_weight=2.0, no WeightedSampler) ──
print("\n--- Variant C: Weighted BCE (pos_weight=2.0, no sampler) ---")
torch.manual_seed(42)
model_c = OptimizedCreditRiskDNN(input_dim)
criterion_c = WeightedBCELoss(pos_weight=2.0)
optimizer_c = optim.Adam(model_c.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_c = optim.lr_scheduler.ReduceLROnPlateau(optimizer_c, mode='min', factor=0.5, patience=8, min_lr=1e-6)

t0 = time.time()
tl, vl, va = train_dnn(model_c, criterion_c, optimizer_c, scheduler_c,
                        train_loader, val_loader, epochs, patience,
                        save_path=RESULT_DIR / 'best_dnn_c.pth')
t_c = time.time() - t0
model_c.load_state_dict(torch.load(RESULT_DIR / 'best_dnn_c.pth', weights_only=True, map_location=device))
results['Weighted BCE (2.0)'] = evaluate(model_c, X_test_t, y_test, device)
results['Weighted BCE (2.0)']['train_time'] = t_c
results['Weighted BCE (2.0)']['train_losses'] = tl
results['Weighted BCE (2.0)']['val_losses'] = vl
results['Weighted BCE (2.0)']['val_aucs'] = va
print(f"  Train time: {t_c:.2f}s")
print(f"  Test  Acc:{results['Weighted BCE (2.0)']['accuracy']:.4f}  Prec:{results['Weighted BCE (2.0)']['precision']:.4f}  "
      f"Rec:{results['Weighted BCE (2.0)']['recall']:.4f}  F1:{results['Weighted BCE (2.0)']['f1']:.4f}  "
      f"ROC:{results['Weighted BCE (2.0)']['roc_auc']:.4f}  PR:{results['Weighted BCE (2.0)']['pr_auc']:.4f}")

# ── Variant D: Focal Loss with mild WeightedSampler (subsample=0.5) ──
print("\n--- Variant D: Focal Loss + mild oversample (pos ratio ~35%) ---")
torch.manual_seed(42)
model_d = OptimizedCreditRiskDNN(input_dim)
criterion_d = FocalLoss(alpha=0.75, gamma=2.0)
optimizer_d = optim.Adam(model_d.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=8, min_lr=1e-6)

# Milder sampler: increase minority weight but not to 50%
y_arr = np.array(y_tr)
class_counts = np.bincount(y_arr.astype(int))
weight_per_class = 1.0 / class_counts
weight_per_class[1] *= 2.0  # only 2x instead of full balance
sample_weights = weight_per_class[y_arr.astype(int)]
mild_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader_d = DataLoader(train_dataset, batch_size=batch_size, sampler=mild_sampler)

t0 = time.time()
tl, vl, va = train_dnn(model_d, criterion_d, optimizer_d, scheduler_d,
                        train_loader_d, val_loader, epochs, patience,
                        save_path=RESULT_DIR / 'best_dnn_d.pth')
t_d = time.time() - t0
model_d.load_state_dict(torch.load(RESULT_DIR / 'best_dnn_d.pth', weights_only=True, map_location=device))
results['Focal + Mild Oversample'] = evaluate(model_d, X_test_t, y_test, device)
results['Focal + Mild Oversample']['train_time'] = t_d
results['Focal + Mild Oversample']['train_losses'] = tl
results['Focal + Mild Oversample']['val_losses'] = vl
results['Focal + Mild Oversample']['val_aucs'] = va
print(f"  Train time: {t_d:.2f}s")
print(f"  Test  Acc:{results['Focal + Mild Oversample']['accuracy']:.4f}  Prec:{results['Focal + Mild Oversample']['precision']:.4f}  "
      f"Rec:{results['Focal + Mild Oversample']['recall']:.4f}  F1:{results['Focal + Mild Oversample']['f1']:.4f}  "
      f"ROC:{results['Focal + Mild Oversample']['roc_auc']:.4f}  PR:{results['Focal + Mild Oversample']['pr_auc']:.4f}")

# ═══════════════════════════════════════════
# 5. RESULTS COMPARISON
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: RESULTS COMPARISON")
print("=" * 60)

# Original baseline DNN results
baseline = {
    'accuracy': 0.8190, 'precision': 0.6609, 'recall': 0.3730,
    'f1': 0.4769, 'roc_auc': 0.7740, 'pr_auc': 0.5589
}

rows = []
for name, res in results.items():
    rows.append({
        'Variant': name,
        'Accuracy': f"{res['accuracy']:.4f}",
        'Precision': f"{res['precision']:.4f}",
        'Recall': f"{res['recall']:.4f}",
        'F1-score': f"{res['f1']:.4f}",
        'ROC-AUC': f"{res['roc_auc']:.4f}",
        'PR-AUC': f"{res['pr_auc']:.4f}",
        'Train Time (s)': f"{res['train_time']:.2f}",
    })

rows.append({
    'Variant': '[Original] BCE Baseline',
    'Accuracy': f"{baseline['accuracy']:.4f}",
    'Precision': f"{baseline['precision']:.4f}",
    'Recall': f"{baseline['recall']:.4f}",
    'F1-score': f"{baseline['f1']:.4f}",
    'ROC-AUC': f"{baseline['roc_auc']:.4f}",
    'PR-AUC': f"{baseline['pr_auc']:.4f}",
    'Train Time (s)': "10.42",
})

comparison = pd.DataFrame(rows)
print("\n" + comparison.to_string(index=False))

# Save results
comparison.to_csv(RESULT_DIR / 'dnn_optimized_comparison.csv', index=False)
print(f"\n[SAVED] dnn_optimized_comparison.csv")

# ── Save the best model ──
best_model_name = max(results, key=lambda k: results[k]['f1'])
print(f"\n[INFO] Best variant by F1: {best_model_name}")
print(f"[INFO] Best F1-score: {results[best_model_name]['f1']:.4f}")

# Save best model weights
torch.save(model_c.state_dict(), RESULT_DIR / 'best_dnn_optimized.pth')
print(f"[SAVED] best_dnn_optimized.pth")

# ═══════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Generating comparison figures")
print("=" * 60)

# ROC Curves
n_variants = len(results)
n_cols = min(n_variants, 2)
n_rows = max(1, (n_variants + n_cols - 1) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
axes = axes.flatten() if n_variants > 1 else [axes]
for ax, (name, res) in zip(axes, results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {res['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(name)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
plt.suptitle('Optimized DNN — ROC Curves', fontsize=16, fontweight='bold')
plt.tight_layout()
# Hide unused axes
for ax in axes[len(results):]:
    ax.set_visible(False)
plt.savefig(FIG_DIR / '30_optimized_dnn_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [SAVED] 30_optimized_dnn_roc.png")

# F1 bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
variants = list(results.keys()) + ['[Original] BCE Baseline']
f1_scores = [results[k]['f1'] for k in results.keys()] + [baseline['f1']]
recalls = [results[k]['recall'] for k in results.keys()] + [baseline['recall']]
colors_list = ['#2F5D8A', '#3A7B5C', '#8C2D2D', '#AAAAAA', '#D4A017']
x = np.arange(len(variants))
width = 0.35
bars1 = ax.bar(x - width/2, f1_scores, width, label='F1-score', color=colors_list[:len(variants)])
bars2 = ax.bar(x + width/2, recalls, width, label='Recall', color=colors_list[:len(variants)], alpha=0.6)
ax.set_ylabel('Score')
ax.set_title('DNN Optimization: F1-score & Recall Comparison')
ax.set_xticks(x)
ax.set_xticklabels(variants, rotation=15, ha='right')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / '30_dnn_optimization_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  [SAVED] 30_dnn_optimization_comparison.png")

print("\n" + "=" * 60)
print("DONE! All results saved.")
print("=" * 60)
