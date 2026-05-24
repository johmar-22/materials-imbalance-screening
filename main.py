
# -*- coding: utf-8 -*-
"""ML_Project_v2_f2

"Overcoming Extreme Class Imbalance in Materials Discovery:
 A Cross-Validated Ablation Study for High-Mobility p-type Semiconductor Screening"
"""

import os
import warnings
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn 
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, auc,
    confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Imbalanced-Learn 
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

# XGBoost 
from xgboost import XGBClassifier

# Ensemble & additional classifiers 
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer


warnings.filterwarnings('ignore')


# 0. PATHS & PLOTTING STYLE

data_dir = 'data'
results_dir = 'results'
figures_dir = 'figures'

os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 11, 'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
})


# 1. DATA LOADING & FEATURE ENGINEERING

print("\n" + "="*60)
print("  LOADING DATA AND ENGINEERING FEATURES")
print("="*60)

df = pd.read_csv(os.path.join(data_dir, 'boltztrap_mp.csv'))

# Primary target: m_p < 1.0 (consistent with p-type TCO literature:
# In2O3 m_p ~ 0.3, CuAlO2 m_p ~ 0.8; Hautier et al. 2013, Nat. Chem.)
TARGET_THRESHOLD = 1.0
df['target'] = (df['m_p'] < TARGET_THRESHOLD).astype(int)

# Physics-safe features: n-type only, avoiding BoltzTraP co-derivation
# leakage (p- and n-type Seebeck/PF are solved from the same BTE).
physical_features = ['m_n', 'PF_n', 'S_n']

# Inject Gaussian noise columns as a positive control for feature selection
np.random.seed(42)
df['Noise_1'] = np.random.normal(0, 1, len(df))
df['Noise_2'] = np.random.normal(0, 1, len(df))
all_features = physical_features + ['Noise_1', 'Noise_2']

X = df[all_features]
y = df['target']

minority_ratio  = y.mean() * 100
imbalance_ratio = (y == 0).sum() / (y == 1).sum()

print(f"Dataset Size:                       {len(df):,} materials")
print(f"High-Mobility minority (m_p<{TARGET_THRESHOLD}):  {y.sum()} ({minority_ratio:.2f}%)")
print(f"Low-Mobility  majority:             {(y==0).sum()} ({100-minority_ratio:.2f}%)")
print(f"Imbalance Ratio (majority:minority): {imbalance_ratio:.1f}:1")
if minority_ratio <= 10:
    print("  -> Class imbalance is EXTREME (minority ratio <= 10%)")
elif minority_ratio <= 20:
    print("  -> Class imbalance is MODERATE-SEVERE (minority ratio <= 20%)")
else:
    print(f"  WARNING: minority ratio {minority_ratio:.1f}% -- consider revising 'extreme' framing.")


# 2. PIPELINE DEFINITIONS

mi_fixed = partial(mutual_info_classif, random_state=42)

pipelines = {
    '1. Full Pipeline': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed, k=3)),
        ('scaler',  StandardScaler()),
        ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('xgb',     XGBClassifier(eval_metric='logloss', random_state=42)),
    ]),
    '2. Ablate SMOTE': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed, k=3)),
        ('scaler',  StandardScaler()),
        ('xgb',     XGBClassifier(eval_metric='logloss', random_state=42)),
    ]),
    '3. Ablate Feat. Select': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('xgb',     XGBClassifier(eval_metric='logloss', random_state=42)),
    ]),
    '4. Ablate XGBoost': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed, k=3)),
        ('scaler',  StandardScaler()),
        ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('dt',      DecisionTreeClassifier(random_state=42)),
    ]),
    '5. Dummy Baseline': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('dummy',   DummyClassifier(strategy='most_frequent')),
    ]),
    '6. RUS Pipeline': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed, k=3)),
        ('scaler',  StandardScaler()),
        ('rus',     RandomUnderSampler(random_state=42, sampling_strategy='auto')),
        ('xgb',     XGBClassifier(eval_metric='logloss', random_state=42)),
    ]),
}

# Hyperparameter search spaces
xgb_param_dist = {
    'xgb__max_depth':     [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'xgb__subsample':     [0.6, 0.8, 1.0],
    'xgb__n_estimators':  [50, 100, 200],
}
dt_param_dist = {
    'dt__max_depth':         [3, 5, 7, 10, None],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf':  [1, 2, 4],
    'dt__criterion':         ['gini', 'entropy'],
}


# 3. NESTED STRATIFIED K-FOLD CROSS-VALIDATION

print("\n" + "="*60)
print("  RUNNING 5-FOLD STRATIFIED CV WITH NESTED HP TUNING")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# F2-Score (beta=2) is the PRIMARY ranking metric -- it weights recall 4x more
# than precision, encoding the asymmetric discovery cost: a missed high-mobility
# candidate (false negative) is costlier than a false alarm (false positive).
# F1 and MCC are retained as secondary metrics for literature comparability.
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score', 'MCC']

oof_preds  = {name: np.zeros(len(X)) for name in pipelines}
oof_probs  = {name: np.zeros(len(X)) for name in pipelines}

# [OVERFITTING CHECK] Train F2-Score and Train Recall are recorded alongside
# held-out (validation) scores. A gap > 0.15 between train and val F2-Score
# is treated as a red flag for overfitting in the diagnostic table below.
fold_metrics = {
    name: {c: [] for c in categories + ['Train F2-Score', 'Train Recall']}
    for name in pipelines
}

feature_importances  = {name: [] for name in pipelines if 'Dummy' not in name}
best_params_history  = {name: [] for name in pipelines if 'Dummy' not in name}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"  Processing Fold {fold + 1}/5...")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, pipeline in pipelines.items():
        if 'Dummy' in name:
            pipeline.fit(X_train, y_train)
            best_model = pipeline

        elif 'dt' in pipeline.named_steps:
            search = RandomizedSearchCV(
                pipeline, dt_param_dist, n_iter=30, cv=3,
                scoring=make_scorer(fbeta_score, beta=2), random_state=42, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params_history[name].append(search.best_params_)

        else:  # XGBoost-based (includes RUS pipeline)
            search = RandomizedSearchCV(
                pipeline, xgb_param_dist, n_iter=30, cv=3,
                scoring=make_scorer(fbeta_score, beta=2), random_state=42, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params_history[name].append(search.best_params_)

        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)[:, 1]
        oof_preds[name][test_idx] = preds
        oof_probs[name][test_idx] = probs

        fold_metrics[name]['Accuracy'].append(accuracy_score(y_test, preds))
        fold_metrics[name]['Precision'].append(precision_score(y_test, preds, zero_division=0))
        fold_metrics[name]['Recall'].append(recall_score(y_test, preds))
        fold_metrics[name]['F1-Score'].append(f1_score(y_test, preds, zero_division=0))
        # F2-Score: beta=2 -> recall weighted 4x more than precision.
        # Formula: (1 + 2^2) * P * R / (2^2 * P + R) = 5PR / (4P + R)
        fold_metrics[name]['F2-Score'].append(fbeta_score(y_test, preds, beta=2, zero_division=0))
        fold_metrics[name]['MCC'].append(matthews_corrcoef(y_test, preds))

        # [OVERFITTING CHECK] Record training-set scores for the same model.
        # Dummy always predicts majority, so its train score is meaningless -- skip.
        if 'Dummy' not in name:
            train_preds = best_model.predict(X_train)
            fold_metrics[name]['Train F2-Score'].append(
                fbeta_score(y_train, train_preds, beta=2, zero_division=0))
            fold_metrics[name]['Train Recall'].append(
                recall_score(y_train, train_preds, zero_division=0))
        else:
            fold_metrics[name]['Train F2-Score'].append(np.nan)
            fold_metrics[name]['Train Recall'].append(np.nan)

        if 'xgb' in best_model.named_steps:
            xgb_step = best_model.named_steps['xgb']
            if 'fs' in best_model.named_steps:
                mask  = best_model.named_steps['fs'].get_support()
                names = np.array(all_features)[mask]
            else:
                names = np.array(all_features)
            feature_importances[name].append(dict(zip(names, xgb_step.feature_importances_)))


# 4. AGGREGATE & PRINT RESULTS

mean_results, fmt_results = [], []
for name in pipelines:
    mean_row = {'Model': name}
    fmt_row  = {'Model': name}
    for c in categories:
        mu  = np.mean(fold_metrics[name][c])
        std = np.std(fold_metrics[name][c])
        mean_row[c] = mu
        fmt_row[c]  = f"{mu:.3f} +/- {std:.3f}"
    mean_results.append(mean_row)
    fmt_results.append(fmt_row)

df_plot = pd.DataFrame(mean_results).set_index('Model')
df_csv  = pd.DataFrame(fmt_results).set_index('Model')

print("\n--- Cross-Validated Ablation Study Results (Mean +/- Std) ---")
print(df_csv.to_string())

# Primary metric: F2-Score
full_f2  = np.mean(fold_metrics['1. Full Pipeline']['F2-Score'])
dummy_f2 = np.mean(fold_metrics['5. Dummy Baseline']['F2-Score'])
rus_f2   = np.mean(fold_metrics['6. RUS Pipeline']['F2-Score'])

# Secondary metrics: F1-Score & MCC
full_f1   = np.mean(fold_metrics['1. Full Pipeline']['F1-Score'])
dummy_f1  = np.mean(fold_metrics['5. Dummy Baseline']['F1-Score'])
rus_f1    = np.mean(fold_metrics['6. RUS Pipeline']['F1-Score'])
full_mcc  = np.mean(fold_metrics['1. Full Pipeline']['MCC'])
dummy_mcc = np.mean(fold_metrics['5. Dummy Baseline']['MCC'])
rus_mcc   = np.mean(fold_metrics['6. RUS Pipeline']['MCC'])

print(f"\n--- Improvement Over Dummy Baseline ---")
print(f"  F2-Score:  Dummy={dummy_f2:.3f}  Full Pipeline={full_f2:.3f}  "
      f"(+{full_f2-dummy_f2:.3f})  RUS={rus_f2:.3f} (+{rus_f2-dummy_f2:.3f})")
print(f"  F1-Score:  Dummy={dummy_f1:.3f}  Full Pipeline={full_f1:.3f}  "
      f"(+{full_f1-dummy_f1:.3f})  RUS={rus_f1:.3f} (+{rus_f1-dummy_f1:.3f})")
print(f"  MCC:       Dummy={dummy_mcc:.3f}  Full Pipeline={full_mcc:.3f}  "
      f"(+{full_mcc-dummy_mcc:.3f})  RUS={rus_mcc:.3f} (+{rus_mcc-dummy_mcc:.3f})")

best_f2_name = max(pipelines.keys(), key=lambda n: np.mean(fold_metrics[n]['F2-Score']))
print(f"\n  Best pipeline by F2-Score (primary metric): {best_f2_name}")

# Save hyperparameters
param_records = []
for name, params_list in best_params_history.items():
    for fold_idx, params in enumerate(params_list):
        row = {'Model': name, 'Fold': fold_idx + 1}
        row.update(params)
        param_records.append(row)
pd.DataFrame(param_records).to_csv(
    os.path.join(results_dir, 'Best_Hyperparameters.csv'), index=False)

df_csv.to_csv(os.path.join(results_dir, 'Ablation_Study_Metrics.csv'))
print(f"\nMetrics saved -> Ablation_Study_Metrics.csv")


# 4b. OVERFITTING / UNDERFITTING DIAGNOSTICS
#
# Three checks:
#   (a) Train vs validation F2-Score gap per pipeline.
#       Gap > 0.15 suggests overfitting to the training fold.
#       Both scores low (< 0.30) suggests underfitting.
#   (b) Per-fold F2-Score standard deviation.
#       Std > 0.08 suggests the model is unstable across splits.
#   (c) Hyperparameter boundary check: if n_estimators was selected as 200
#       (the top of the search range) in >= 4 out of 5 folds, the model may
#       need a wider search range and could be underfitting on estimators.

print("\n" + "="*60)
print("  OVERFITTING / UNDERFITTING DIAGNOSTICS")
print("="*60)

# (a) Train vs validation F2-Score gap
print("\n[a] Train vs Validation F2-Score Gap")
print(f"    Threshold: gap > 0.15 = OVERFIT warning | val F2 < 0.30 = UNDERFIT warning\n")
print(f"  {'Model':<30} {'Train F2':>9} {'Val F2':>9} {'Gap':>8}  Status")
print("  " + "-"*65)

gap_records = []
for name in pipelines:
    train_scores = [s for s in fold_metrics[name]['Train F2-Score'] if not np.isnan(s)]
    val_scores   = fold_metrics[name]['F2-Score']
    if not train_scores:
        continue
    mean_train = np.mean(train_scores)
    mean_val   = np.mean(val_scores)
    gap        = mean_train - mean_val
    if gap > 0.15:
        status = "OVERFIT?"
    elif mean_val < 0.30 and mean_train < 0.30:
        status = "UNDERFIT?"
    else:
        status = "OK"
    print(f"  {name:<30} {mean_train:>9.3f} {mean_val:>9.3f} {gap:>8.3f}  {status}")
    gap_records.append({
        'Pipeline': name, 'Mean Train F2': round(mean_train, 4),
        'Mean Val F2': round(mean_val, 4), 'Gap': round(gap, 4), 'Status': status,
    })

df_gap = pd.DataFrame(gap_records)
df_gap.to_csv(os.path.join(results_dir, 'Overfitting_TrainVal_Gap.csv'), index=False)
print(f"\n  Saved: Overfitting_TrainVal_Gap.csv")

# (b) Per-fold F2-Score variance
print("\n[b] Per-Fold F2-Score Stability (Std > 0.08 = UNSTABLE warning)\n")
print(f"  {'Model':<30} {'Std':>7}  {'Fold scores':>45}  Status")
print("  " + "-"*100)

var_records = []
for name in pipelines:
    scores = fold_metrics[name]['F2-Score']
    std    = np.std(scores)
    status = "UNSTABLE?" if std > 0.08 else "OK"
    fold_str = str([round(s, 3) for s in scores])
    print(f"  {name:<30} {std:>7.3f}  {fold_str:>45}  {status}")
    var_records.append({
        'Pipeline': name, 'Std F2': round(std, 4),
        **{f'Fold {i+1} F2': round(scores[i], 4) for i in range(len(scores))},
        'Status': status,
    })

df_var = pd.DataFrame(var_records)
df_var.to_csv(os.path.join(results_dir, 'Overfitting_FoldVariance.csv'), index=False)
print(f"\n  Saved: Overfitting_FoldVariance.csv")

# (c) Hyperparameter boundary check for n_estimators
print("\n[c] Hyperparameter Boundary Check (n_estimators)")
print("    If n_estimators = 200 in >= 4/5 folds, search range may be too narrow.\n")
_N_EST_MAX = max(xgb_param_dist['xgb__n_estimators'])
for name, params_list in best_params_history.items():
    if not params_list:
        continue
    n_est_vals = [p.get('xgb__n_estimators', None) for p in params_list]
    n_est_vals = [v for v in n_est_vals if v is not None]
    if not n_est_vals:
        continue
    at_boundary = sum(v == _N_EST_MAX for v in n_est_vals)
    flag = "RANGE WARNING" if at_boundary >= 4 else "OK"
    print(f"  {name:<30}  n_estimators selected: {n_est_vals}  [{flag}]")


# 5. FEATURE IMPORTANCE AGGREGATION

full_imp_records = feature_importances.get('3. Ablate Feat. Select', [])
if full_imp_records:
    imp_df   = pd.DataFrame(full_imp_records).fillna(0)
    mean_imp = imp_df.mean().sort_values(ascending=False)
    std_imp  = imp_df.std().fillna(0)
    imp_summary = pd.DataFrame({'Mean Importance': mean_imp, 'Std': std_imp})
    print("\n--- Feature Importances (Ablate Feat. Select pipeline, Mean +/- Std) ---")
    print(imp_summary.round(4))
    imp_summary.to_csv(os.path.join(results_dir, 'Feature_Importances.csv'))


# 6. VISUALISATIONS

print("\nGenerating Publication-Ready Plots...")

colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba', '#808080', '#7b2d8b']

# PLOT 1: Confusion Matrices
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
})

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Ablation Study: Cross-Validated Confusion Matrices\n'
    r'(Target: High-Mobility p-type Semiconductors, $m_p < 1.0\,m_e$)',
    fontsize=20, fontweight='bold', y=1.03,
)
axes = axes.flatten()

for i, name in enumerate(pipelines):
    if i >= 6:
        break
    cm = confusion_matrix(y, oof_preds[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                annot_kws={"size": 22, "weight": "bold"},
                linewidths=1, linecolor='black')
    oof_f2 = fbeta_score(y, oof_preds[name], beta=2, zero_division=0)
    axes[i].set_title(f"{name}\n(F2={oof_f2:.3f})", fontweight='bold', pad=12)
    axes[i].set_xlabel('Predicted Class', fontweight='bold', labelpad=8)
    axes[i].set_ylabel('Actual Class', fontweight='bold', labelpad=8)
    axes[i].set_xticklabels(['Low-Mob.','High-Mob.'], rotation=0)
    axes[i].set_yticklabels(['Low-Mob.','High-Mob.'], rotation=90, va='center')

plt.tight_layout()
p = os.path.join(figures_dir, 'Fig1_Confusion_Matrices.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
})

# PLOT 2: ROC & PR curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for name, color in zip(pipelines, colors):
    fpr, tpr, _ = roc_curve(y, oof_probs[name])
    roc_auc     = auc(fpr, tpr)
    ls = '--' if 'Dummy' in name else '-'
    ax1.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
             label=f'{name} (AUC={roc_auc:.2f})')
    prec, rec, _ = precision_recall_curve(y, oof_probs[name])
    ap = average_precision_score(y, oof_probs[name])
    ax2.plot(rec, prec, color=color, lw=2, linestyle=ls,
             label=f'{name} (AP={ap:.2f})')

ax1.plot([0,1],[0,1], 'k:', lw=1.5, label='Random')
ax1.set(xlim=[0,1], ylim=[0,1.05], xlabel='False Positive Rate',
        ylabel='True Positive Rate', title='Receiver Operating Characteristic (ROC)')
ax1.legend(loc='lower right', fontsize=9)

baseline = y.mean()
ax2.axhline(baseline, color='k', lw=1.5, linestyle=':', label=f'No-skill ({baseline:.3f})')
ax2.set(xlim=[0,1], ylim=[0,1.05], xlabel='Recall (Sensitivity)',
        ylabel='Precision (PPV)', title='Precision-Recall (PR) Curve')
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
p = os.path.join(figures_dir, 'Fig2_ROC_PR_Curves.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

# PLOT 3: Radar Chart
# F2-Score added as the sixth axis; MCC is normalised to [0,1] via (MCC+1)/2.
df_radar = df_plot.copy()
df_radar['MCC'] = (df_radar['MCC'] + 1) / 2
df_radar.rename(columns={'MCC': 'MCC (Norm.)'}, inplace=True)
cats_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score', 'MCC (Norm.)']
N      = len(cats_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], cats_radar, size=12, fontweight='bold')
plt.ylim(0.0, 1.0)
plt.yticks([0.2,0.4,0.6,0.8,1.0], ["0.2","0.4","0.6","0.8","1.0"],
           color="grey", size=10)

for (idx, row), color in zip(df_radar.iterrows(), colors):
    vals  = row[cats_radar].values.flatten().tolist() + \
            [row[cats_radar].values[0]]
    ls    = '--' if 'Dummy' in idx else '-'
    lw    = 1.5 if 'Dummy' in idx else 2.5
    alpha = 0.05 if 'Dummy' in idx else 0.10
    ax.plot(angles, vals, lw=lw, linestyle=ls, label=idx, color=color)
    ax.fill(angles, vals, color=color, alpha=alpha)

plt.legend(loc='upper right', bbox_to_anchor=(0.12, 0.12),
           prop={'size': 9, 'weight': 'bold'})
plt.title("Cross-Validated Ablation Impact on Performance Metrics\n"
          r"(MCC normalised: $(MCC+1)/2$; F2-Score: $\beta=2$; dashed = Dummy Baseline)",
          size=12, weight='bold', y=1.12)
plt.tight_layout()
p = os.path.join(figures_dir, 'Fig3_Radar_Chart.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

# PLOT 4: Feature Importance
if full_imp_records:
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_colors = ['#2b83ba' if f in physical_features else '#cccccc'
                  for f in mean_imp.index]
    bars = ax.bar(mean_imp.index, mean_imp.values,
                  yerr=std_imp[mean_imp.index].values,
                  color=imp_colors, edgecolor='black', linewidth=0.8,
                  capsize=5, error_kw={'linewidth': 1.5})
    ax.set_xlabel('Feature', fontweight='bold')
    ax.set_ylabel('Mean XGBoost Feature Importance\n(across 5 CV folds)', fontweight='bold')
    ax.set_title('Feature Importances: Ablated Feature Selection Pipeline\n'
                 '(Blue = Physical Features, Grey = Injected Noise)', fontweight='bold')
    ax.set_ylim(0, min(1.0, mean_imp.max() * 1.4))
    for bar, val in zip(bars, mean_imp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(figures_dir, 'Fig4_Feature_Importances.pdf')
    plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  Saved: {p}")


# 6b. LEARNING CURVE (Fig8) -- Pipeline 1 (Full Pipeline)
#
# Purpose: Determine whether the model is in the high-bias (underfitting) or
# high-variance (overfitting) regime. Uses fixed representative hyperparameters
# to avoid running nested RandomizedSearchCV at every training set size, which
# would be computationally prohibitive. The hyperparameters (max_depth=5,
# learning_rate=0.1, n_estimators=100, subsample=0.8) are representative
# values within the search space; they are used here purely for diagnostic
# purposes and do not replace the tuned models used in Section 3.
#
# Interpretation guide (printed below the figure):
#   - Large persistent gap at full training size -> overfitting
#   - Both curves converge at a low value         -> underfitting
#   - Both curves converge at a high value        -> well-fitted
#   - Val score still rising at the rightmost point -> more data would help

print("\n" + "="*60)
print("  LEARNING CURVE DIAGNOSTIC (Fig8)")
print("  Pipeline 1 -- Full Pipeline, fixed representative HPs")
print("  Primary metric: F2-Score (beta=2)")
print("="*60)

f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

# Fixed-HP version of Pipeline 1 used only for the learning curve diagnostic.
# These are not the tuned hyperparameters -- they are representative values
# chosen to make the learning curve computationally feasible.
lc_pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('fs',      SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=3)),
    ('scaler',  StandardScaler()),
    ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
    ('xgb',     XGBClassifier(
                    eval_metric='logloss',
                    max_depth=5, learning_rate=0.1,
                    n_estimators=100, subsample=0.8,
                    random_state=42, verbosity=0)),
])

# train_sizes starts at 0.2 (not 0.1) because at 10% of ~7228 training samples
# (~723 samples, ~36 minority), Borderline-SMOTE can fail on some folds.
train_sizes_frac = np.linspace(0.2, 1.0, 9)

print("  Running learning_curve (this may take several minutes)...")
train_sizes_abs, train_scores_lc, val_scores_lc = learning_curve(
    estimator   = lc_pipeline,
    X           = X,
    y           = y,
    cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring     = f2_scorer,
    train_sizes = train_sizes_frac,
    n_jobs      = -1,
    error_score = np.nan,
)

train_mean = np.nanmean(train_scores_lc, axis=1)
train_std  = np.nanstd(train_scores_lc,  axis=1)
val_mean   = np.nanmean(val_scores_lc,   axis=1)
val_std    = np.nanstd(val_scores_lc,    axis=1)

# Diagnosis
final_gap = train_mean[-1] - val_mean[-1]
still_rising = (val_mean[-1] - val_mean[-2]) > 0.01

if final_gap > 0.15:
    lc_diagnosis = "OVERFIT: large train/val gap persists at full training size."
elif val_mean[-1] < 0.30 and train_mean[-1] < 0.30:
    lc_diagnosis = "UNDERFIT: both curves converge at a low F2-Score."
elif still_rising:
    lc_diagnosis = "IMPROVING: validation score still rising -- more data may help further."
else:
    lc_diagnosis = "WELL-FITTED: train/val curves converge at a reasonable F2-Score."

print(f"\n  Final train F2: {train_mean[-1]:.3f} +/- {train_std[-1]:.3f}")
print(f"  Final val   F2: {val_mean[-1]:.3f}  +/- {val_std[-1]:.3f}")
print(f"  Train/val gap:  {final_gap:.3f}")
print(f"  Diagnosis:      {lc_diagnosis}")

fig, ax = plt.subplots(figsize=(9, 6))

ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color='#d7191c')
ax.fill_between(train_sizes_abs, val_mean   - val_std,   val_mean   + val_std,
                alpha=0.15, color='#2b83ba')
ax.plot(train_sizes_abs, train_mean, 'o-', color='#d7191c', lw=2,
        label='Training F2-Score')
ax.plot(train_sizes_abs, val_mean,   's-', color='#2b83ba', lw=2,
        label='Validation F2-Score (5-fold CV)')

ax.set_xlabel('Training Set Size (samples)', fontweight='bold')
ax.set_ylabel(r'F2-Score ($\beta=2$)', fontweight='bold')
ax.set_title(
    'Learning Curve: Pipeline 1 (Full Pipeline)\n'
    r'Borderline-SMOTE + SelectKBest + XGBoost  |  Primary metric: F2-Score ($\beta=2$)',
    fontweight='bold',
)
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True, linestyle=':', alpha=0.6)
ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=11)

# Annotate the gap at the rightmost point
ax.annotate(
    f'Gap = {final_gap:.3f}\n({lc_diagnosis.split(":")[0]})',
    xy=(train_sizes_abs[-1], (train_mean[-1] + val_mean[-1]) / 2),
    xytext=(train_sizes_abs[-1] * 0.72, (train_mean[-1] + val_mean[-1]) / 2 + 0.08),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
    fontsize=10, color='black',
)

plt.tight_layout()
p = os.path.join(figures_dir, 'Fig8_Learning_Curve.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

# Save learning curve data to CSV for reporting
lc_df = pd.DataFrame({
    'Train Size':   train_sizes_abs,
    'Train F2 Mean': train_mean.round(4),
    'Train F2 Std':  train_std.round(4),
    'Val F2 Mean':   val_mean.round(4),
    'Val F2 Std':    val_std.round(4),
    'Gap':          (train_mean - val_mean).round(4),
})
lc_df.to_csv(os.path.join(results_dir, 'Learning_Curve_Data.csv'), index=False)
print(f"  Saved: Learning_Curve_Data.csv")


# 7. PCA "CRIME SCENE" PLOT

print("\n  Generating PCA Crime Scene plot (SMOTE diagnostic)...")

imputer_pca = SimpleImputer(strategy='median')
scaler_pca  = StandardScaler()
X_phys = X[physical_features].copy()
X_phys_imp   = imputer_pca.fit_transform(X_phys)
X_phys_scale = scaler_pca.fit_transform(X_phys_imp)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_phys_scale)
var_exp = pca.explained_variance_ratio_ * 100

bsmote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_res, y_res = bsmote.fit_resample(X_phys_scale, y)

n_real      = len(X_phys_scale)
n_synthetic = len(X_res) - n_real
X_synth_scale = X_res[n_real:]
X_synth_2d    = pca.transform(X_synth_scale)

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 1.5

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

maj_mask = y.values == 0
min_mask = y.values == 1

ax.scatter(X_2d[maj_mask, 0], X_2d[maj_mask, 1],
           c='#5A5A5A', s=20, alpha=0.6, edgecolor='none',
           label=f'Real Majority (n={maj_mask.sum():,})', zorder=1)
ax.scatter(X_2d[min_mask, 0], X_2d[min_mask, 1],
           c='#1976D2', s=50, alpha=0.9, edgecolor='white', linewidths=0.6,
           label=f'Real Minority (n={min_mask.sum():,})', zorder=2)
ax.scatter(X_synth_2d[:, 0], X_synth_2d[:, 1],
           c='#D32F2F', marker='X', s=45, edgecolor='white', linewidths=0.3, alpha=0.8,
           label=f'Synthetic (Borderline-SMOTE, n={n_synthetic:,})', zorder=3)

ax.set_xlabel(f'Principal Component 1 ({var_exp[0]:.1f}% variance)', fontweight='bold', labelpad=10)
ax.set_ylabel(f'Principal Component 2 ({var_exp[1]:.1f}% variance)', fontweight='bold', labelpad=10)
ax.set_title('PCA Feature Space: Real Data vs. SMOTE Hallucinations', fontsize=15, fontweight='bold', pad=20)

leg = ax.legend(loc='upper left', frameon=True, framealpha=1.0, edgecolor='black',
                fontsize=11, title="Material Class", borderpad=1)
leg.get_title().set_fontweight('bold')

axins = ax.inset_axes([0.52, 0.25, 0.45, 0.45])
axins.scatter(X_2d[maj_mask, 0], X_2d[maj_mask, 1], c='#5A5A5A', s=15, alpha=0.6, edgecolor='none', zorder=1)
axins.scatter(X_2d[min_mask, 0], X_2d[min_mask, 1], c='#1976D2', s=45, alpha=0.9, edgecolor='white', linewidths=0.5, zorder=2)
axins.scatter(X_synth_2d[:, 0], X_synth_2d[:, 1], c='#D32F2F', marker='X', s=35, edgecolor='white', linewidths=0.3, alpha=0.8, zorder=3)

x1, x2, y1, y2 = -1.5, 2.5, -1.5, 2.5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_title("Magnified Phase Boundary", fontsize=12, fontweight='bold', pad=8)
axins.tick_params(labelsize=10)
axins.grid(True, linestyle=':', alpha=0.7, color='gray')
for spine in axins.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)

ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=2, alpha=0.6)
sns.despine(ax=ax)

plt.tight_layout()
p = os.path.join(figures_dir, 'Fig5_PCA_SMOTE_CrimeScene.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")



# 9. ENSEMBLE EXTENSION NO RESAMPLING (Pipelines 7-10)

print("\n" + "="*60)
print("  ENSEMBLE EXTENSION NO RESAMPLING ")
print("="*60)

mi_fixed_k3 = partial(mutual_info_classif, random_state=42)

extra_pipelines = {
    '7. Random Forest': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        #('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     RandomForestClassifier(
                        n_estimators=200,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1,
        )),
    ]),
    '8. Logistic Reg.': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        #('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     LogisticRegression(
                        C=1.0,
                        solver='lbfgs',
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=42,
        )),
    ]),
    '9. SVM (RBF)': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        #('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     SVC(
                        kernel='rbf',
                        C=1.0,
                        gamma='scale',
                        probability=True,
                        class_weight='balanced',
                        random_state=42,
        )),
    ]),
    '10. Stacking Ensemble': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        #('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     StackingClassifier(
            estimators=[
                ('xgb', XGBClassifier(
                    eval_metric='logloss', n_estimators=100,
                    max_depth=5, random_state=42, verbosity=0)),
                ('rf',  RandomForestClassifier(
                    n_estimators=100, class_weight='balanced',
                    random_state=42, n_jobs=-1)),
                ('svm', SVC(
                    kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42)),
            ],
            final_estimator=LogisticRegression(
                C=1.0, solver='lbfgs', class_weight='balanced',
                max_iter=1000, random_state=42),
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1,
        )),
    ]),
}

extra_fold_metrics = {
    name: {'Accuracy': [], 'Precision': [], 'Recall': [],
           'F1-Score': [], 'F2-Score': [], 'MCC': []}
    for name in extra_pipelines
}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"  Ensemble fold {fold + 1}/5...")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, pipe in extra_pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        extra_fold_metrics[name]['Accuracy'].append(accuracy_score(y_test, preds))
        extra_fold_metrics[name]['Precision'].append(
            precision_score(y_test, preds, zero_division=0))
        extra_fold_metrics[name]['Recall'].append(recall_score(y_test, preds))
        extra_fold_metrics[name]['F1-Score'].append(f1_score(y_test, preds, zero_division=0))
        extra_fold_metrics[name]['F2-Score'].append(
            fbeta_score(y_test, preds, beta=2, zero_division=0))
        extra_fold_metrics[name]['MCC'].append(matthews_corrcoef(y_test, preds))

extra_mean_results, extra_fmt_results = [], []
for name in extra_pipelines:
    mean_row = {'Model': name}
    fmt_row  = {'Model': name}
    for c in categories:
        mu  = np.mean(extra_fold_metrics[name][c])
        std = np.std(extra_fold_metrics[name][c])
        mean_row[c] = mu
        fmt_row[c]  = f"{mu:.3f} +/- {std:.3f}"
    extra_mean_results.append(mean_row)
    extra_fmt_results.append(fmt_row)

df_extra_plot = pd.DataFrame(extra_mean_results).set_index('Model')
df_extra_csv  = pd.DataFrame(extra_fmt_results).set_index('Model')

print("\n--- Extra Pipeline Results (Mean +/- Std, 5-fold CV) NO RESAMPLING ---")
print(df_extra_csv.to_string())

df_extra_csv.to_csv(os.path.join(results_dir, 'Extra_Pipeline_Results_NO RESAMPLING.csv'))
print(f"\n  Saved: Extra_Pipeline_Results_NO RESAMPLING.csv")

# Figure 7: Nature-quality grouped bar chart -- F1, F2 (primary), MCC
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

extra_names_clean = [
    'Random\nforest',
    'Logistic\nregression',
    'SVM\n(RBF kernel)',
    'Stacking\nensemble',
]

f1_means  = np.array([np.mean(extra_fold_metrics[n]['F1-Score']) for n in extra_pipelines])
f1_stds   = np.array([np.std( extra_fold_metrics[n]['F1-Score']) for n in extra_pipelines])
f2_means  = np.array([np.mean(extra_fold_metrics[n]['F2-Score']) for n in extra_pipelines])
f2_stds   = np.array([np.std( extra_fold_metrics[n]['F2-Score']) for n in extra_pipelines])
mcc_means = np.array([np.mean(extra_fold_metrics[n]['MCC'])      for n in extra_pipelines])
mcc_stds  = np.array([np.std( extra_fold_metrics[n]['MCC'])      for n in extra_pipelines])

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.frameon':    False,
    'legend.fontsize':   8,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

FIG_W = 183 / 25.4
FIG_H = 70  / 25.4

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

n_groups = len(extra_names_clean)
x        = np.arange(n_groups)
w        = 0.22
gap      = 0.03

COL_F1  = '#1B4F72'
COL_F2  = '#1E8449'
COL_MCC = '#B03A2E'

ECAP = dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor='#444444', zorder=4)

bars_f1 = ax.bar(
    x - w - gap, f1_means, w,
    color=COL_F1, linewidth=0, zorder=3, label='F1-score (secondary)')
ax.errorbar(x - w - gap, f1_means, yerr=f1_stds, fmt='none', **ECAP)

bars_f2 = ax.bar(
    x, f2_means, w,
    color=COL_F2, linewidth=0, zorder=3, label=r'F2-score $(\beta=2)$ -- PRIMARY')
ax.errorbar(x, f2_means, yerr=f2_stds, fmt='none', **ECAP)

bars_mcc = ax.bar(
    x + w + gap, mcc_means, w,
    color=COL_MCC, linewidth=0, zorder=3, label='MCC (secondary)')
ax.errorbar(x + w + gap, mcc_means, yerr=mcc_stds, fmt='none', **ECAP)

for bar, val in zip(bars_f1, f1_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#222222')
for bar, val in zip(bars_f2, f2_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#222222')
for bar, val in zip(bars_mcc, mcc_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#222222')

ax.set_xticks(x)
ax.set_xticklabels(extra_names_clean, multialignment='center')
ax.set_ylabel('Score')
ax.set_ylim(0, max(f1_means.max(), f2_means.max(), mcc_means.max()) +
               max(f1_stds.max(), f2_stds.max(), mcc_stds.max()) + 0.25)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.yaxis.grid(True, linewidth=0.4, color='#cccccc', zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis='x', length=0)

legend_patches = [
    mpatches.Patch(color=COL_F1,  label='F1-score'),
    mpatches.Patch(color=COL_F2,  label=r'F2-score'),
    mpatches.Patch(color=COL_MCC, label='MCC'),
]
ax.legend(
    handles=legend_patches,
    loc='upper right',
    ncol=1,
    handlelength=1.2,
    handleheight=0.8,
    borderpad=0.4,
    labelspacing=0.3,
    handletextpad=0.5,
)

plt.tight_layout(pad=0.5)
p = os.path.join(figures_dir, 'Fig7_Ensemble_Comparison_NO RESAMPLING.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
p_tiff = os.path.join(figures_dir, 'Fig7_Ensemble_Comparison_NO RESAMPLING.tiff')
plt.savefig(p_tiff, dpi=300, bbox_inches='tight', format='tiff')
plt.close()
print(f"  Saved: {p}")
print(f"  Saved: {p_tiff}")

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 11,
    'axes.spines.top': True, 'axes.spines.right': True,
})




# 9. ENSEMBLE EXTENSION  (Pipelines 7-10)

print("\n" + "="*60)
print("  ENSEMBLE EXTENSION ")
print("="*60)

mi_fixed_k3 = partial(mutual_info_classif, random_state=42)

extra_pipelines = {
    '7. Random Forest': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     RandomForestClassifier(
                        n_estimators=200,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1,
        )),
    ]),
    '8. Logistic Reg.': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     LogisticRegression(
                        C=1.0,
                        solver='lbfgs',
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=42,
        )),
    ]),
    '9. SVM (RBF)': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     SVC(
                        kernel='rbf',
                        C=1.0,
                        gamma='scale',
                        probability=True,
                        class_weight='balanced',
                        random_state=42,
        )),
    ]),
    '10. Stacking Ensemble': ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('fs',      SelectKBest(score_func=mi_fixed_k3, k=3)),
        ('scaler',  StandardScaler()),
        ('smote',   BorderlineSMOTE(random_state=42, kind='borderline-1')),
        ('clf',     StackingClassifier(
            estimators=[
                ('xgb', XGBClassifier(
                    eval_metric='logloss', n_estimators=100,
                    max_depth=5, random_state=42, verbosity=0)),
                ('rf',  RandomForestClassifier(
                    n_estimators=100, class_weight='balanced',
                    random_state=42, n_jobs=-1)),
                ('svm', SVC(
                    kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42)),
            ],
            final_estimator=LogisticRegression(
                C=1.0, solver='lbfgs', class_weight='balanced',
                max_iter=1000, random_state=42),
            cv=3,
            stack_method='predict_proba',
            n_jobs=-1,
        )),
    ]),
}

extra_fold_metrics = {
    name: {'Accuracy': [], 'Precision': [], 'Recall': [],
           'F1-Score': [], 'F2-Score': [], 'MCC': []}
    for name in extra_pipelines
}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"  Ensemble fold {fold + 1}/5...")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, pipe in extra_pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        extra_fold_metrics[name]['Accuracy'].append(accuracy_score(y_test, preds))
        extra_fold_metrics[name]['Precision'].append(
            precision_score(y_test, preds, zero_division=0))
        extra_fold_metrics[name]['Recall'].append(recall_score(y_test, preds))
        extra_fold_metrics[name]['F1-Score'].append(f1_score(y_test, preds, zero_division=0))
        extra_fold_metrics[name]['F2-Score'].append(
            fbeta_score(y_test, preds, beta=2, zero_division=0))
        extra_fold_metrics[name]['MCC'].append(matthews_corrcoef(y_test, preds))

extra_mean_results, extra_fmt_results = [], []
for name in extra_pipelines:
    mean_row = {'Model': name}
    fmt_row  = {'Model': name}
    for c in categories:
        mu  = np.mean(extra_fold_metrics[name][c])
        std = np.std(extra_fold_metrics[name][c])
        mean_row[c] = mu
        fmt_row[c]  = f"{mu:.3f} +/- {std:.3f}"
    extra_mean_results.append(mean_row)
    extra_fmt_results.append(fmt_row)

df_extra_plot = pd.DataFrame(extra_mean_results).set_index('Model')
df_extra_csv  = pd.DataFrame(extra_fmt_results).set_index('Model')

print("\n--- Extra Pipeline Results (Mean +/- Std, 5-fold CV) ---")
print(df_extra_csv.to_string())

df_extra_csv.to_csv(os.path.join(results_dir, 'Extra_Pipeline_Results.csv'))
print(f"\n  Saved: Extra_Pipeline_Results.csv")

# Figure 7: Nature-quality grouped bar chart -- F1, F2 (primary), MCC
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

extra_names_clean = [
    'Random\nforest',
    'Logistic\nregression',
    'SVM\n(RBF kernel)',
    'Stacking\nensemble',
]

f1_means  = np.array([np.mean(extra_fold_metrics[n]['F1-Score']) for n in extra_pipelines])
f1_stds   = np.array([np.std( extra_fold_metrics[n]['F1-Score']) for n in extra_pipelines])
f2_means  = np.array([np.mean(extra_fold_metrics[n]['F2-Score']) for n in extra_pipelines])
f2_stds   = np.array([np.std( extra_fold_metrics[n]['F2-Score']) for n in extra_pipelines])
mcc_means = np.array([np.mean(extra_fold_metrics[n]['MCC'])      for n in extra_pipelines])
mcc_stds  = np.array([np.std( extra_fold_metrics[n]['MCC'])      for n in extra_pipelines])

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'legend.frameon':    False,
    'legend.fontsize':   8,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

FIG_W = 183 / 25.4
FIG_H = 70  / 25.4

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

n_groups = len(extra_names_clean)
x        = np.arange(n_groups)
w        = 0.22
gap      = 0.03

COL_F1  = '#1B4F72'
COL_F2  = '#1E8449'
COL_MCC = '#B03A2E'

ECAP = dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor='#444444', zorder=4)

bars_f1 = ax.bar(
    x - w - gap, f1_means, w,
    color=COL_F1, linewidth=0, zorder=3, label='F1-score (secondary)')
ax.errorbar(x - w - gap, f1_means, yerr=f1_stds, fmt='none', **ECAP)

bars_f2 = ax.bar(
    x, f2_means, w,
    color=COL_F2, linewidth=0, zorder=3, label=r'F2-score $(\beta=2)$ -- PRIMARY')
ax.errorbar(x, f2_means, yerr=f2_stds, fmt='none', **ECAP)

bars_mcc = ax.bar(
    x + w + gap, mcc_means, w,
    color=COL_MCC, linewidth=0, zorder=3, label='MCC (secondary)')
ax.errorbar(x + w + gap, mcc_means, yerr=mcc_stds, fmt='none', **ECAP)

for bar, val in zip(bars_f1, f1_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#222222')
for bar, val in zip(bars_f2, f2_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#222222')
for bar, val in zip(bars_mcc, mcc_means):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
            f'{val:.3f}', ha='center', va='bottom', fontsize=5.5, color='#222222')

ax.set_xticks(x)
ax.set_xticklabels(extra_names_clean, multialignment='center')
ax.set_ylabel('Score')
ax.set_ylim(0, max(f1_means.max(), f2_means.max(), mcc_means.max()) +
               max(f1_stds.max(), f2_stds.max(), mcc_stds.max()) + 0.25)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.yaxis.grid(True, linewidth=0.4, color='#cccccc', zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis='x', length=0)

legend_patches = [
    mpatches.Patch(color=COL_F1,  label='F1-score'),
    mpatches.Patch(color=COL_F2,  label=r'F2-score'),
    mpatches.Patch(color=COL_MCC, label='MCC'),
]
ax.legend(
    handles=legend_patches,
    loc='upper right',
    ncol=1,
    handlelength=1.2,
    handleheight=0.8,
    borderpad=0.4,
    labelspacing=0.3,
    handletextpad=0.5,
)

plt.tight_layout(pad=0.5)
p = os.path.join(figures_dir, 'Fig7_Ensemble_Comparison.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
p_tiff = os.path.join(figures_dir, 'Fig7_Ensemble_Comparison.tiff')
plt.savefig(p_tiff, dpi=300, bbox_inches='tight', format='tiff')
plt.close()
print(f"  Saved: {p}")
print(f"  Saved: {p_tiff}")

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 11,
    'axes.spines.top': True, 'axes.spines.right': True,
})



# 11. SUMMARY

print("\n" + "="*60)
print("  ALL OUTPUTS SAVED SUCCESSFULLY")
print("="*60)
print(f"\nOutput directories: {results_dir} and {figures_dir}\n")
print("  Ablation_Study_Metrics.csv        -- Main ablation table (Pipelines 1-6), includes F2-Score")
print("  Best_Hyperparameters.csv          -- Per-fold best HP history")
print("  Feature_Importances.csv           -- XGBoost feature importance")
print("  Threshold_Sensitivity.csv         -- m_p < 0.5/1.0/2.0 comparison (F1, F2, MCC)")
print("  Extra_Pipeline_Results.csv        -- Ensemble extension (Pipelines 7-10), includes F2-Score")
print("  Overfitting_TrainVal_Gap.csv      -- [DIAGNOSTIC] Train vs val F2-Score gap per pipeline")
print("  Overfitting_FoldVariance.csv      -- [DIAGNOSTIC] Per-fold F2-Score std per pipeline")
print("  Learning_Curve_Data.csv           -- [DIAGNOSTIC] Learning curve data for Pipeline 1")
print("  Fig1_Confusion_Matrices.pdf       -- 6-panel confusion matrices (with F2 in subtitle)")
print("  Fig2_ROC_PR_Curves.pdf            -- ROC and PR curves (6 pipelines)")
print("  Fig3_Radar_Chart.pdf              -- Multi-metric radar chart (6 axes incl. F2-Score)")
print("  Fig4_Feature_Importances.pdf      -- Feature importance bar chart")
print("  Fig5_PCA_SMOTE_CrimeScene.pdf     -- PCA diagnostic: real vs synthetic")
print("  Fig6_Threshold_Sensitivity.pdf    -- Threshold robustness (3-panel: F1, F2, MCC)")
print("  Fig7_Ensemble_Comparison.pdf      -- F1, F2 (primary), MCC for Pipelines 7-10")
print("  Fig8_Learning_Curve.pdf           -- [DIAGNOSTIC] Learning curve for Pipeline 1")
print("  LLM_Explanations.txt              -- API explanations (if key was set)")
print("\nDone.")

# -*- coding: utf-8 -*-
"""
Statistical Pipeline Comparison  (v3 -- F2-Score as primary metric)
===================================================================
Two complementary tests for comparing ML pipelines evaluated under
stratified k-fold cross-validation:

  1. McNemar's test on OOF predictions (Dietterich 1998, Neural Computation)
     - High statistical power -- uses all N samples
     - Holm-Bonferroni correction for multiple comparisons
     - METRIC-AGNOSTIC: always tests binary correct/incorrect disagreements.
       Your F2-Score preference determines which pipeline is declared the
       winner when the test is significant; it does not change the test
       statistic itself.

  2. Wilcoxon signed-rank test on per-fold F2-Score (PRIMARY)
     and per-fold F1-Score, Recall, MCC (SECONDARY)
     (Demsar 2006, JMLR)
     - Directly respects the CV evaluation protocol
     - Non-parametric -- no normality assumption
     - F2-Score (beta=2) is used as the primary test metric, consistent
       with the decision made before running any experiments.
     - LIMITATION: low statistical power at n_folds=5. Non-significance
       does not mean equivalence -- interpret cautiously.

Usage
-----
Call run_all_tests(oof_preds, fold_metrics, y, save_dir=results_dir)
from your main notebook cell after Section 3 has populated oof_preds,
fold_metrics, and y.

References
----------
Dietterich T.G. (1998). Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms. Neural Computation 10(7).

Demsar J. (2006). Statistical Comparisons of Classifiers over Multiple
  Data Sets. JMLR 7, 1-30.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


# ============================================================================
# INTERNAL UTILITY
# ============================================================================

def _holm_bonferroni(p_raw_series):
    """
    Holm-Bonferroni step-down correction.

    Parameters
    ----------
    p_raw_series : list of raw p-values, already sorted ascending.

    Returns
    -------
    list of corrected p-values, monotone non-decreasing, each capped at 1.0.

    Notes
    -----
    Standard Holm (1979): adjusted p_(i) = min(1, (m - i + 1) * p_(i)).
    Monotonicity is enforced by taking the running maximum so that
    p_adj_(i) >= p_adj_(i-1) for all i. Without this step the procedure
    is mathematically invalid (a higher-ranked pair could receive a smaller
    adjusted p than a lower-ranked one).
    """
    m = len(p_raw_series)
    adjusted = [min(1.0, p_raw_series[i] * (m - i)) for i in range(m)]
    # enforce non-decreasing (running maximum)
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    return [round(v, 6) for v in adjusted]


# ============================================================================
# 1. MCNEMAR'S TEST
# ============================================================================

def mcnemar_pairwise(oof_preds: dict, y: np.ndarray,
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    McNemar's test (continuity-corrected) for every pipeline pair.
    Holm-Bonferroni correction applied across all pairs.

    Parameters
    ----------
    oof_preds : dict  {pipeline_name -> np.ndarray of shape (N,)}
    y         : array-like of shape (N,), true labels
    alpha     : family-wise error rate

    Returns
    -------
    pd.DataFrame sorted by raw p-value (ascending).

    Notes
    -----
    The 'winner' column indicates which pipeline made fewer errors overall
    (larger c means A beats B). This is independent of F2-Score -- McNemar
    always operates on binary correctness. Cross-reference with mean
    F2-Score to confirm the winner is also preferred under your metric.
    Valid when b+c >= 10; a warning is printed otherwise.
    """
    y = np.asarray(y)
    names   = list(oof_preds.keys())
    pairs   = list(itertools.combinations(names, 2))
    rows    = []

    for name_a, name_b in pairs:
        preds_a   = np.asarray(oof_preds[name_a])
        preds_b   = np.asarray(oof_preds[name_b])
        correct_a = (preds_a == y)
        correct_b = (preds_b == y)

        # b: A wrong, B right  |  c: A right, B wrong
        b = int(np.sum(~correct_a &  correct_b))
        c = int(np.sum( correct_a & ~correct_b))

        table = np.array([
            [int(np.sum( correct_a &  correct_b)), c],
            [b,                                    int(np.sum(~correct_a & ~correct_b))],
        ])

        bc_sum = b + c
        note   = (f"WARNING: b+c={bc_sum} < 10; result unreliable."
                  if bc_sum < 10 else "")

        result = mcnemar(table, exact=False, correction=True)

        if c > b:
            winner = name_a
        elif b > c:
            winner = name_b
        else:
            winner = "tie"

        rows.append({
            'Pipeline_A':           name_a,
            'Pipeline_B':           name_b,
            'b (A wrong, B right)': b,
            'c (A right, B wrong)': c,
            'b+c':                  bc_sum,
            'statistic':            round(result.statistic, 4),
            'p_raw':                result.pvalue,
            'winner':               winner,
            'note':                 note,
        })

    df = pd.DataFrame(rows).sort_values('p_raw').reset_index(drop=True)
    df['p_corrected_holm'] = _holm_bonferroni(df['p_raw'].tolist())
    df['significant']      = df['p_corrected_holm'] < alpha

    return df


# ============================================================================
# 2. WILCOXON SIGNED-RANK TEST
# ============================================================================

def wilcoxon_pairwise(fold_metrics: dict,
                      metric: str  = 'F2-Score',
                      alpha: float = 0.05) -> pd.DataFrame:
    """
    Wilcoxon signed-rank test for every pipeline pair on a given metric.
    Holm-Bonferroni correction applied.

    Parameters
    ----------
    fold_metrics : dict  {pipeline_name -> {metric_name -> [fold_scores]}}
    metric       : which metric to test. Default 'F2-Score' (primary).
                   Also call with 'F1-Score', 'Recall', 'MCC'.
    alpha        : family-wise error rate

    Returns
    -------
    pd.DataFrame sorted by raw p-value (ascending).

    Notes
    -----
    At n=5 folds the minimum achievable p-value is 0.0625, so significance
    at alpha=0.05 is impossible regardless of the true effect size. Report
    this limitation explicitly. Non-significance does not confirm equivalence.
    """
    names  = list(fold_metrics.keys())
    pairs  = list(itertools.combinations(names, 2))
    rows   = []

    for name_a, name_b in pairs:
        scores_a = np.array(fold_metrics[name_a][metric])
        scores_b = np.array(fold_metrics[name_b][metric])
        diff     = scores_a - scores_b

        if np.all(diff == 0):
            stat, p_raw = np.nan, 1.0
            note = "All fold differences zero -- cannot perform test."
        else:
            try:
                stat, p_raw = wilcoxon(scores_a, scores_b,
                                       alternative='two-sided',
                                       zero_method='wilcox')
                note = ""
            except ValueError as e:
                stat, p_raw = np.nan, 1.0
                note = f"Test failed: {e}"

        mean_a = round(float(np.mean(scores_a)), 4)
        mean_b = round(float(np.mean(scores_b)), 4)

        if mean_a > mean_b:
            winner = name_a
        elif mean_b > mean_a:
            winner = name_b
        else:
            winner = "tie"

        rows.append({
            'Pipeline_A':        name_a,
            'Pipeline_B':        name_b,
            f'mean_{metric}_A':  mean_a,
            f'mean_{metric}_B':  mean_b,
            'statistic':         round(stat, 4) if not np.isnan(stat) else np.nan,
            'p_raw':             p_raw,
            'winner':            winner,
            'note':              note,
        })

    df = pd.DataFrame(rows).sort_values('p_raw').reset_index(drop=True)
    df['p_corrected_holm'] = _holm_bonferroni(df['p_raw'].tolist())
    df['significant']      = df['p_corrected_holm'] < alpha

    return df


# ============================================================================
# 3. WIN SUMMARY TABLE
# ============================================================================

def pipeline_win_summary(mcnemar_df: pd.DataFrame,
                         wilcoxon_f2_df: pd.DataFrame,
                         wilcoxon_secondary: dict,
                         mean_f2: dict) -> pd.DataFrame:
    """
    Build a ranked summary of statistically significant wins per pipeline.

    A win is counted only when:
      - The comparison is significant after Holm correction, AND
      - That pipeline is the declared winner.

    Parameters
    ----------
    mcnemar_df         : output of mcnemar_pairwise()
    wilcoxon_f2_df     : output of wilcoxon_pairwise() with metric='F2-Score'
    wilcoxon_secondary : dict {metric_name -> wilcoxon_pairwise() DataFrame}
                         for F1-Score, Recall, MCC
    mean_f2            : dict {pipeline_name -> mean F2-Score across folds}

    Returns
    -------
    pd.DataFrame ranked by: F2 wins > McNemar wins > Mean F2-Score.
    """
    from collections import defaultdict

    wins = defaultdict(lambda: {
        'mcnemar_wins': 0, 'f2_wins': 0,
        'f1_wins': 0, 'recall_wins': 0, 'mcc_wins': 0,
    })

    all_names = set()
    for df_ in [mcnemar_df, wilcoxon_f2_df] + list(wilcoxon_secondary.values()):
        all_names.update(df_['Pipeline_A'].tolist())
        all_names.update(df_['Pipeline_B'].tolist())

    for _, row in mcnemar_df[mcnemar_df['significant']].iterrows():
        if row['winner'] != 'tie':
            wins[row['winner']]['mcnemar_wins'] += 1

    for _, row in wilcoxon_f2_df[wilcoxon_f2_df['significant']].iterrows():
        if row['winner'] != 'tie':
            wins[row['winner']]['f2_wins'] += 1

    secondary_key_map = {
        'F1-Score': 'f1_wins',
        'Recall':   'recall_wins',
        'MCC':      'mcc_wins',
    }
    for metric, df_ in wilcoxon_secondary.items():
        key = secondary_key_map.get(metric, metric + '_wins')
        for _, row in df_[df_['significant']].iterrows():
            if row['winner'] != 'tie':
                wins[row['winner']][key] += 1

    records = []
    for name in all_names:
        w = wins[name]
        records.append({
            'Pipeline':             name,
            'F2 Wilcoxon wins':     w['f2_wins'],
            'McNemar wins':         w['mcnemar_wins'],
            'F1 Wilcoxon wins':     w['f1_wins'],
            'Recall Wilcoxon wins': w['recall_wins'],
            'MCC Wilcoxon wins':    w['mcc_wins'],
            'Mean F2-Score':        round(mean_f2.get(name, np.nan), 4),
        })

    df_summary = (
        pd.DataFrame(records)
        .sort_values(
            ['F2 Wilcoxon wins', 'McNemar wins', 'Mean F2-Score'],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    df_summary.insert(0, 'Rank', range(1, len(df_summary) + 1))
    return df_summary


# ============================================================================
# 4. MAIN RUNNER
# ============================================================================

def run_all_tests(oof_preds: dict,
                  fold_metrics: dict,
                  y,
                  save_dir: str = '.',
                  alpha: float = 0.05):
    """
    Run all statistical comparisons and save results to CSV.

    Primary test : Wilcoxon on per-fold F2-Score
    Secondary    : Wilcoxon on per-fold F1-Score, Recall, MCC
    High-power   : McNemar on OOF binary predictions (N=9036)

    Parameters
    ----------
    oof_preds    : {pipeline_name -> OOF predicted label array (N,)}
    fold_metrics : {pipeline_name -> {metric_name -> [per-fold scores]}}
                   Must contain 'F2-Score', 'F1-Score', 'Recall', 'MCC'.
    y            : true labels, shape (N,)
    save_dir     : directory to write CSV output files
    alpha        : family-wise significance threshold (default 0.05)

    Returns
    -------
    tuple of (mcnemar_df, wilcoxon_f2_df, secondary_dfs, summary_df)
    """
    import os
    y = np.asarray(y)

    required_metrics = ['F2-Score', 'F1-Score', 'Recall', 'MCC']
    for name, mdict in fold_metrics.items():
        for m in required_metrics:
            if m not in mdict:
                raise KeyError(
                    f"Pipeline '{name}' missing metric '{m}' in fold_metrics. "
                    f"Found: {list(mdict.keys())}"
                )

    print("\n" + "=" * 60)
    print("  SECTION 11: STATISTICAL PIPELINE COMPARISON")
    print("  Primary metric : F2-Score (beta=2)")
    print("  Secondary      : F1-Score, Recall, MCC")
    print("=" * 60)

    # -- McNemar --------------------------------------------------------------
    print("\n[1/5] McNemar's Test (OOF predictions, Holm-Bonferroni corrected)")
    print("      Continuity-corrected chi-squared (Dietterich 1998)")
    print("      Winner = pipeline with fewer overall errors (cross-reference")
    print("      with F2-Score to confirm preferred direction).\n")

    df_mcnemar = mcnemar_pairwise(oof_preds, y, alpha=alpha)

    cols_mcn = ['Pipeline_A', 'Pipeline_B',
                'b (A wrong, B right)', 'c (A right, B wrong)',
                'b+c', 'statistic', 'p_raw',
                'p_corrected_holm', 'significant', 'winner']
    print(df_mcnemar[cols_mcn].to_string(index=False))
    for _, row in df_mcnemar.iterrows():
        if row['note']:
            print(f"  ! {row['Pipeline_A']} vs {row['Pipeline_B']}: {row['note']}")

    df_mcnemar.to_csv(os.path.join(save_dir, 'Stat_McNemar_Results.csv'), index=False)
    print("\n  Saved: Stat_McNemar_Results.csv")

    # -- Wilcoxon PRIMARY: F2-Score -------------------------------------------
    n_folds = len(list(fold_metrics.values())[0]['F2-Score'])
    print("\n[2/5] Wilcoxon Signed-Rank Test -- PRIMARY metric: F2-Score (Demsar 2006)")
    print(f"      n_folds={n_folds}; minimum achievable p=0.0625 -- significance")
    print("      at alpha=0.05 is not possible. Non-significance does NOT")
    print("      confirm equivalence. Report this limitation explicitly.\n")

    df_wil_f2 = wilcoxon_pairwise(fold_metrics, metric='F2-Score', alpha=alpha)

    cols_wil = ['Pipeline_A', 'Pipeline_B',
                'mean_F2-Score_A', 'mean_F2-Score_B',
                'statistic', 'p_raw', 'p_corrected_holm', 'significant', 'winner']
    print(df_wil_f2[cols_wil].to_string(index=False))
    for _, row in df_wil_f2.iterrows():
        if row['note']:
            print(f"  ! {row['Pipeline_A']} vs {row['Pipeline_B']}: {row['note']}")

    df_wil_f2.to_csv(os.path.join(save_dir, 'Stat_Wilcoxon_F2_Results.csv'), index=False)
    print("\n  Saved: Stat_Wilcoxon_F2_Results.csv")

    # -- Wilcoxon SECONDARY: F1, Recall, MCC ----------------------------------
    secondary_dfs     = {}
    secondary_metrics = ['F1-Score', 'Recall', 'MCC']

    for i, metric in enumerate(secondary_metrics, start=3):
        print(f"\n[{i}/5] Wilcoxon Signed-Rank Test -- secondary metric: {metric}\n")
        df_sec = wilcoxon_pairwise(fold_metrics, metric=metric, alpha=alpha)

        mean_col_a = f'mean_{metric}_A'
        mean_col_b = f'mean_{metric}_B'
        cols_sec   = ['Pipeline_A', 'Pipeline_B',
                      mean_col_a, mean_col_b,
                      'statistic', 'p_raw', 'p_corrected_holm', 'significant', 'winner']
        print(df_sec[cols_sec].to_string(index=False))
        for _, row in df_sec.iterrows():
            if row['note']:
                print(f"  ! {row['Pipeline_A']} vs {row['Pipeline_B']}: {row['note']}")

        fname = f'Stat_Wilcoxon_{metric.replace("-", "").replace(" ", "_")}_Results.csv'
        df_sec.to_csv(os.path.join(save_dir, fname), index=False)
        print(f"\n  Saved: {fname}")
        secondary_dfs[metric] = df_sec

    # -- Summary --------------------------------------------------------------
    print(f"\n[5/5] Win Summary -- significant wins only "
          f"(Holm-corrected, alpha={alpha})")
    print("      Ranking: F2 Wilcoxon wins > McNemar wins > Mean F2-Score\n")

    mean_f2_lookup = {
        name: float(np.mean(scores['F2-Score']))
        for name, scores in fold_metrics.items()
    }

    df_summary = pipeline_win_summary(
        df_mcnemar, df_wil_f2, secondary_dfs, mean_f2_lookup
    )
    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(save_dir, 'Stat_Pipeline_Ranking.csv'), index=False)
    print("\n  Saved: Stat_Pipeline_Ranking.csv")

    best        = df_summary.iloc[0]['Pipeline']
    best_f2     = df_summary.iloc[0]['Mean F2-Score']
    best_f2_wins = df_summary.iloc[0]['F2 Wilcoxon wins']
    print(f"\n  Recommended pipeline : {best}")
    print(f"  Mean F2-Score        : {best_f2:.4f}")
    print(f"  Significant F2 wins  : {best_f2_wins}")
    print("  (Confirm against Wilcoxon and McNemar tables above before reporting.)")

    return df_mcnemar, df_wil_f2, secondary_dfs, df_summary

run_all_tests(oof_preds, fold_metrics, y, save_dir=results_dir)
