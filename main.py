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
from sklearn.metrics import make_scorer
# Imbalanced-Learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
# XGBoost
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# 0. PATHS & PLOTTING STYLE
data_dir    = 'data'
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

# 1b. DATA CLEANING
# The six BoltzTraP transport features used for target definition and modelling.
SIX_FEATURES = ['m_n', 'PF_n', 'S_n', 'm_p', 'PF_p', 'S_p']
n_raw = len(df)

# (i) Duplicate mpid check — keep first occurrence
if 'mpid' in df.columns:
    n_dup = df.duplicated(subset='mpid').sum()
    if n_dup > 0:
        print(f"  [Cleaning] Removing {n_dup} duplicate mpid row(s).")
        df = df.drop_duplicates(subset='mpid', keep='first').reset_index(drop=True)
    else:
        print(f"  [Cleaning] No duplicate mpid entries found.")
else:
    print("  [Cleaning] Column 'mpid' not found — skipping duplicate check.")

# (ii) NaN check across the six transport features — drop incomplete rows
n_nan_rows = df[SIX_FEATURES].isnull().any(axis=1).sum()
if n_nan_rows > 0:
    print(f"  [Cleaning] Removing {n_nan_rows} row(s) with NaN in transport features.")
    df = df.dropna(subset=SIX_FEATURES).reset_index(drop=True)
else:
    print(f"  [Cleaning] No NaN values found in the six transport features.")

n_retained = len(df)
print(f"  [Cleaning] Raw: {n_raw:,}  |  Retained: {n_retained:,}  "
      f"|  Removed: {n_raw - n_retained:,}")

# Positive class: m_p < 1.0 m_e — the dispersive-valence-band prerequisite
# for hole mobility (Hautier et al. 2013, Nat. Commun.; Wang et al. 2024).
# This is a necessary but not sufficient condition for realized p-type
# conductivity; confirmed p-type character additionally requires bandgap,
# dopability, and stability screening beyond BoltzTraP-derived quantities.
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
    print(f"  WARNING: minority ratio {minority_ratio:.1f}% -- consider revising framing.")

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
# is treated as a red flag for overfitting.
fold_metrics = {
    name: {c: [] for c in categories + ['Train F2-Score', 'Train Recall']}
    for name in pipelines
}
feature_importances = {name: [] for name in pipelines if 'Dummy' not in name}
best_params_history = {name: [] for name in pipelines if 'Dummy' not in name}

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
        fold_metrics[name]['F2-Score'].append(fbeta_score(y_test, preds, beta=2, zero_division=0))
        fold_metrics[name]['MCC'].append(matthews_corrcoef(y_test, preds))

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

full_f2  = np.mean(fold_metrics['1. Full Pipeline']['F2-Score'])
dummy_f2 = np.mean(fold_metrics['5. Dummy Baseline']['F2-Score'])
rus_f2   = np.mean(fold_metrics['6. RUS Pipeline']['F2-Score'])
full_f1  = np.mean(fold_metrics['1. Full Pipeline']['F1-Score'])
dummy_f1 = np.mean(fold_metrics['5. Dummy Baseline']['F1-Score'])
rus_f1   = np.mean(fold_metrics['6. RUS Pipeline']['F1-Score'])
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

# FIGURE 3 (paper): Confusion Matrices
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
    axes[i].set_xticklabels(['Low-Mob.', 'High-Mob.'], rotation=0)
    axes[i].set_yticklabels(['Low-Mob.', 'High-Mob.'], rotation=90, va='center')
plt.tight_layout()
p = os.path.join(figures_dir, 'Fig3_Confusion_Matrices.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
})

# FIGURE 4 (paper): ROC & PR curves
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
ax1.plot([0, 1], [0, 1], 'k:', lw=1.5, label='Random')
ax1.set(xlim=[0, 1], ylim=[0, 1.05], xlabel='False Positive Rate',
        ylabel='True Positive Rate', title='Receiver Operating Characteristic (ROC)')
ax1.legend(loc='lower right', fontsize=9)
baseline = y.mean()
ax2.axhline(baseline, color='k', lw=1.5, linestyle=':', label=f'No-skill ({baseline:.3f})')
ax2.set(xlim=[0, 1], ylim=[0, 1.05], xlabel='Recall (Sensitivity)',
        ylabel='Precision (PPV)', title='Precision-Recall (PR) Curve')
ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout()
p = os.path.join(figures_dir, 'Fig4_ROC_PR_Curves.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

# FIGURE S1 (supplementary): Radar Chart
# MCC normalised to [0,1] via (MCC+1)/2; F2-Score added as sixth axis.
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
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"],
           color="grey", size=10)
for (idx, row), color in zip(df_radar.iterrows(), colors):
    vals  = row[cats_radar].values.flatten().tolist() + [row[cats_radar].values[0]]
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
p = os.path.join(figures_dir, 'FigS1_Radar_Chart.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

# FIGURE 5 (paper): Feature Importances
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
    p = os.path.join(figures_dir, 'Fig5_Feature_Importances.pdf')
    plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  Saved: {p}")

# FIGURE S2 (supplementary): Learning Curve — Pipeline 1 (Full Pipeline)
#
# Diagnostic only — not included in main paper.
# Uses fixed representative hyperparameters to avoid running nested
# RandomizedSearchCV at every training set size.
print("\n" + "="*60)
print("  LEARNING CURVE DIAGNOSTIC (FigS2)")
print("  Pipeline 1 -- Full Pipeline, fixed representative HPs")
print("  Primary metric: F2-Score (beta=2)")
print("="*60)

f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

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

final_gap   = train_mean[-1] - val_mean[-1]
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
ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                alpha=0.15, color='#2b83ba')
ax.plot(train_sizes_abs, train_mean, 'o-', color='#d7191c', lw=2,
        label='Training F2-Score')
ax.plot(train_sizes_abs, val_mean, 's-', color='#2b83ba', lw=2,
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
ax.annotate(
    f'Gap = {final_gap:.3f}\n({lc_diagnosis.split(":")[0]})',
    xy=(train_sizes_abs[-1], (train_mean[-1] + val_mean[-1]) / 2),
    xytext=(train_sizes_abs[-1] * 0.72, (train_mean[-1] + val_mean[-1]) / 2 + 0.08),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
    fontsize=10, color='black',
)
plt.tight_layout()
p = os.path.join(figures_dir, 'FigS2_Learning_Curve.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

lc_df = pd.DataFrame({
    'Train Size':    train_sizes_abs,
    'Train F2 Mean': train_mean.round(4),
    'Train F2 Std':  train_std.round(4),
    'Val F2 Mean':   val_mean.round(4),
    'Val F2 Std':    val_std.round(4),
    'Gap':           (train_mean - val_mean).round(4),
})
lc_df.to_csv(os.path.join(results_dir, 'Learning_Curve_Data.csv'), index=False)
print(f"  Saved: Learning_Curve_Data.csv")

# FIGURE 6 (paper): PCA Feature Space
print("\n  Generating PCA feature space plot...")
imputer_pca = SimpleImputer(strategy='median')
scaler_pca  = StandardScaler()
X_phys = X[physical_features].copy()
X_phys_imp   = imputer_pca.fit_transform(X_phys)
X_phys_scale = scaler_pca.fit_transform(X_phys_imp)

pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_phys_scale)
var_exp = pca.explained_variance_ratio_ * 100

bsmote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_res, y_res = bsmote.fit_resample(X_phys_scale, y)
n_real        = len(X_phys_scale)
n_synthetic   = len(X_res) - n_real
X_synth_scale = X_res[n_real:]
X_synth_2d    = pca.transform(X_synth_scale)

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth']   = 1.5

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

ax.set_xlabel(f'Principal Component 1 ({var_exp[0]:.1f}% variance)',
              fontweight='bold', labelpad=10)
ax.set_ylabel(f'Principal Component 2 ({var_exp[1]:.1f}% variance)',
              fontweight='bold', labelpad=10)
ax.set_title('PCA Projection of n-type Feature Space\n'
             r'($m_n$, $PF_n$, $S_n$) — Real vs. Borderline-SMOTE Synthetic Samples',
             fontsize=15, fontweight='bold', pad=20)

leg = ax.legend(loc='upper left', frameon=True, framealpha=1.0, edgecolor='black',
                fontsize=11, title="Material Class", borderpad=1)
leg.get_title().set_fontweight('bold')

axins = ax.inset_axes([0.52, 0.25, 0.45, 0.45])
axins.scatter(X_2d[maj_mask, 0], X_2d[maj_mask, 1],
              c='#5A5A5A', s=15, alpha=0.6, edgecolor='none', zorder=1)
axins.scatter(X_2d[min_mask, 0], X_2d[min_mask, 1],
              c='#1976D2', s=45, alpha=0.9, edgecolor='white', linewidths=0.5, zorder=2)
axins.scatter(X_synth_2d[:, 0], X_synth_2d[:, 1],
              c='#D32F2F', marker='X', s=35, edgecolor='white', linewidths=0.3, alpha=0.8, zorder=3)
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
p = os.path.join(figures_dir, 'Fig6_PCA_Feature_Space.pdf')
plt.savefig(p, dpi=600, bbox_inches='tight', format='pdf')
plt.close()
print(f"  Saved: {p}")

# 7. STATISTICAL PIPELINE COMPARISON
import itertools
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test


def _holm_bonferroni(p_raw_series):
    m        = len(p_raw_series)
    adjusted = [min(1.0, p_raw_series[i] * (m - i)) for i in range(m)]
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    return [round(v, 6) for v in adjusted]


def mcnemar_pairwise(oof_preds, y, alpha=0.05):
    y     = np.asarray(y)
    names = list(oof_preds.keys())
    pairs = list(itertools.combinations(names, 2))
    rows  = []
    for name_a, name_b in pairs:
        preds_a   = np.asarray(oof_preds[name_a])
        preds_b   = np.asarray(oof_preds[name_b])
        correct_a = (preds_a == y)
        correct_b = (preds_b == y)
        b = int(np.sum(~correct_a &  correct_b))
        c = int(np.sum( correct_a & ~correct_b))
        table = np.array([
            [int(np.sum( correct_a &  correct_b)), c],
            [b, int(np.sum(~correct_a & ~correct_b))],
        ])
        bc_sum = b + c
        note   = (f"WARNING: b+c={bc_sum} < 10; result unreliable."
                  if bc_sum < 10 else "")
        result = mcnemar_test(table, exact=False, correction=True)
        winner = name_a if c > b else (name_b if b > c else "tie")
        rows.append({
            'Pipeline_A': name_a, 'Pipeline_B': name_b,
            'b (A wrong, B right)': b, 'c (A right, B wrong)': c,
            'b+c': bc_sum, 'statistic': round(result.statistic, 4),
            'p_raw': result.pvalue, 'winner': winner, 'note': note,
        })
    df = pd.DataFrame(rows).sort_values('p_raw').reset_index(drop=True)
    df['p_corrected_holm'] = _holm_bonferroni(df['p_raw'].tolist())
    df['significant']      = df['p_corrected_holm'] < alpha
    return df


def wilcoxon_pairwise(fold_metrics, metric='F2-Score', alpha=0.05):
    names = list(fold_metrics.keys())
    pairs = list(itertools.combinations(names, 2))
    rows  = []
    for name_a, name_b in pairs:
        scores_a = np.array(fold_metrics[name_a][metric])
        scores_b = np.array(fold_metrics[name_b][metric])
        diff     = scores_a - scores_b
        if np.all(diff == 0):
            stat, p_raw, note = np.nan, 1.0, "All fold differences zero."
        else:
            try:
                stat, p_raw = wilcoxon(scores_a, scores_b,
                                       alternative='two-sided', zero_method='wilcox')
                note = ""
            except ValueError as e:
                stat, p_raw, note = np.nan, 1.0, f"Test failed: {e}"
        mean_a = round(float(np.mean(scores_a)), 4)
        mean_b = round(float(np.mean(scores_b)), 4)
        winner = name_a if mean_a > mean_b else (name_b if mean_b > mean_a else "tie")
        rows.append({
            'Pipeline_A': name_a, 'Pipeline_B': name_b,
            f'mean_{metric}_A': mean_a, f'mean_{metric}_B': mean_b,
            'statistic': round(stat, 4) if not np.isnan(stat) else np.nan,
            'p_raw': p_raw, 'winner': winner, 'note': note,
        })
    df = pd.DataFrame(rows).sort_values('p_raw').reset_index(drop=True)
    df['p_corrected_holm'] = _holm_bonferroni(df['p_raw'].tolist())
    df['significant']      = df['p_corrected_holm'] < alpha
    return df


def pipeline_win_summary(mcnemar_df, wilcoxon_f2_df, wilcoxon_secondary, mean_f2):
    from collections import defaultdict
    wins      = defaultdict(lambda: {'mcnemar_wins': 0, 'f2_wins': 0,
                                     'f1_wins': 0, 'recall_wins': 0, 'mcc_wins': 0})
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
    key_map = {'F1-Score': 'f1_wins', 'Recall': 'recall_wins', 'MCC': 'mcc_wins'}
    for metric, df_ in wilcoxon_secondary.items():
        key = key_map.get(metric, metric + '_wins')
        for _, row in df_[df_['significant']].iterrows():
            if row['winner'] != 'tie':
                wins[row['winner']][key] += 1
    records = [{'Pipeline': n,
                'F2 Wilcoxon wins':     wins[n]['f2_wins'],
                'McNemar wins':         wins[n]['mcnemar_wins'],
                'F1 Wilcoxon wins':     wins[n]['f1_wins'],
                'Recall Wilcoxon wins': wins[n]['recall_wins'],
                'MCC Wilcoxon wins':    wins[n]['mcc_wins'],
                'Mean F2-Score':        round(mean_f2.get(n, np.nan), 4)}
               for n in all_names]
    df_out = (pd.DataFrame(records)
              .sort_values(['F2 Wilcoxon wins', 'McNemar wins', 'Mean F2-Score'],
                           ascending=False)
              .reset_index(drop=True))
    df_out.insert(0, 'Rank', range(1, len(df_out) + 1))
    return df_out


def run_all_tests(oof_preds, fold_metrics, y, save_dir='.', alpha=0.05):
    y = np.asarray(y)
    required = ['F2-Score', 'F1-Score', 'Recall', 'MCC']
    for name, mdict in fold_metrics.items():
        for m in required:
            if m not in mdict:
                raise KeyError(f"Pipeline '{name}' missing metric '{m}'.")

    print("\n" + "="*60)
    print("  STATISTICAL PIPELINE COMPARISON")
    print("  Primary metric : F2-Score (beta=2)")
    print("  Secondary      : F1-Score, Recall, MCC")
    print("="*60)

    print("\n[1/5] McNemar's Test (OOF predictions, Holm-Bonferroni corrected)\n")
    df_mcnemar = mcnemar_pairwise(oof_preds, y, alpha=alpha)
    cols_mcn = ['Pipeline_A', 'Pipeline_B',
                'b (A wrong, B right)', 'c (A right, B wrong)',
                'b+c', 'statistic', 'p_raw', 'p_corrected_holm', 'significant', 'winner']
    print(df_mcnemar[cols_mcn].to_string(index=False))
    df_mcnemar.to_csv(os.path.join(save_dir, 'Stat_McNemar_Results.csv'), index=False)
    print("\n  Saved: Stat_McNemar_Results.csv")

    n_folds = len(list(fold_metrics.values())[0]['F2-Score'])
    print(f"\n[2/5] Wilcoxon Signed-Rank -- PRIMARY: F2-Score")
    print(f"      n_folds={n_folds}; min achievable p=0.0625; non-significance != equivalence\n")
    df_wil_f2 = wilcoxon_pairwise(fold_metrics, metric='F2-Score', alpha=alpha)
    cols_wil  = ['Pipeline_A', 'Pipeline_B',
                 'mean_F2-Score_A', 'mean_F2-Score_B',
                 'statistic', 'p_raw', 'p_corrected_holm', 'significant', 'winner']
    print(df_wil_f2[cols_wil].to_string(index=False))
    df_wil_f2.to_csv(os.path.join(save_dir, 'Stat_Wilcoxon_F2_Results.csv'), index=False)
    print("\n  Saved: Stat_Wilcoxon_F2_Results.csv")

    secondary_dfs = {}
    for i, metric in enumerate(['F1-Score', 'Recall', 'MCC'], start=3):
        print(f"\n[{i}/5] Wilcoxon -- secondary: {metric}\n")
        df_sec = wilcoxon_pairwise(fold_metrics, metric=metric, alpha=alpha)
        mean_col_a = f'mean_{metric}_A'
        mean_col_b = f'mean_{metric}_B'
        cols_sec   = ['Pipeline_A', 'Pipeline_B', mean_col_a, mean_col_b,
                      'statistic', 'p_raw', 'p_corrected_holm', 'significant', 'winner']
        print(df_sec[cols_sec].to_string(index=False))
        fname = f'Stat_Wilcoxon_{metric.replace("-","").replace(" ","_")}_Results.csv'
        df_sec.to_csv(os.path.join(save_dir, fname), index=False)
        print(f"\n  Saved: {fname}")
        secondary_dfs[metric] = df_sec

    print(f"\n[5/5] Win Summary (Holm-corrected, alpha={alpha})\n")
    mean_f2_lookup = {n: float(np.mean(s['F2-Score'])) for n, s in fold_metrics.items()}
    df_summary = pipeline_win_summary(df_mcnemar, df_wil_f2, secondary_dfs, mean_f2_lookup)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(os.path.join(save_dir, 'Stat_Pipeline_Ranking.csv'), index=False)
    print("\n  Saved: Stat_Pipeline_Ranking.csv")

    best = df_summary.iloc[0]
    print(f"\n  Recommended pipeline : {best['Pipeline']}")
    print(f"  Mean F2-Score        : {best['Mean F2-Score']:.4f}")
    print(f"  Significant F2 wins  : {best['F2 Wilcoxon wins']}")

    return df_mcnemar, df_wil_f2, secondary_dfs, df_summary


run_all_tests(oof_preds, fold_metrics, y, save_dir=results_dir)

# 8. SUMMARY
print("\n" + "="*60)
print("  ALL OUTPUTS SAVED SUCCESSFULLY")
print("="*60)
print(f"\nOutput directories: {results_dir}/ and {figures_dir}/\n")
print("  Results files:")
print("    Ablation_Study_Metrics.csv          -- Main ablation table (Pipelines 1-6)")
print("    Best_Hyperparameters.csv            -- Per-fold best HP history")
print("    Feature_Importances.csv             -- XGBoost MDI feature importance")
print("    Overfitting_TrainVal_Gap.csv         -- [Diagnostic] Train vs val F2 gap")
print("    Overfitting_FoldVariance.csv         -- [Diagnostic] Per-fold F2 std")
print("    Learning_Curve_Data.csv              -- [Diagnostic] Learning curve data")
print("    Stat_McNemar_Results.csv             -- McNemar pairwise test")
print("    Stat_Wilcoxon_F2_Results.csv         -- Wilcoxon test (primary: F2)")
print("    Stat_Wilcoxon_F1Score_Results.csv    -- Wilcoxon test (secondary: F1)")
print("    Stat_Wilcoxon_Recall_Results.csv     -- Wilcoxon test (secondary: Recall)")
print("    Stat_Wilcoxon_MCC_Results.csv        -- Wilcoxon test (secondary: MCC)")
print("    Stat_Pipeline_Ranking.csv            -- Win summary table")
print("\n  Figure files (numbered to match paper):")
print("    Fig3_Confusion_Matrices.pdf          -- 6-panel confusion matrices")
print("    Fig4_ROC_PR_Curves.pdf               -- ROC and PR curves")
print("    Fig5_Feature_Importances.pdf         -- XGBoost feature importance")
print("    Fig6_PCA_Feature_Space.pdf           -- PCA projection (real vs synthetic)")
print("    FigS1_Radar_Chart.pdf                -- [Supplementary] Multi-metric radar")
print("    FigS2_Learning_Curve.pdf             -- [Supplementary] Learning curve")
print("\nDone.")
