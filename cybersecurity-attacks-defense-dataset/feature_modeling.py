# -*- coding: utf-8 -*-
"""
Feature Engineering & Modeling - Cybersecurity Datasets
4 tasks:
  1. CVE Ransomware Prediction (Binary)
  2. OTX Primary Industry Classification (Multi-Class)
  3. Malicious Domain Threat Severity (Multi-Class)
  4. Malicious IP Threat Severity (Multi-Class)

Models: LogisticRegression, RandomForest, XGBoost, LightGBM
Evaluation: Stratified K-Fold CV, classification reports, confusion matrices, feature importance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              f1_score, accuracy_score, precision_score, recall_score,
                              RocCurveDisplay, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

BASE_DIR = r"E:\py_project\Cybersecurity Attacks & Defense Dataset 2026"
CLEAN_DIR = os.path.join(BASE_DIR, "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "model_outputs")
os.makedirs(MODEL_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11, "savefig.bbox": "tight",
})
sns.set_style("whitegrid")
PALETTE = "Set2"

RANDOM_STATE = 42
N_SPLITS = 5

# ============================================================================
# Load & Prepare
# ============================================================================
print("Loading cleaned datasets...")

otx = pd.read_csv(os.path.join(CLEAN_DIR, "1_otx_threat_intel_clean.csv"))
cve = pd.read_csv(os.path.join(CLEAN_DIR, "2_cve_vulnerabilities_clean.csv"))
domains = pd.read_csv(os.path.join(CLEAN_DIR, "3_malicious_domains_clean.csv"))
ips = pd.read_csv(os.path.join(CLEAN_DIR, "4_malicious_ips_clean.csv"))

def parse_list(val):
    if pd.isna(val) or val == "[]" or val == "":
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [x.strip() for x in str(val).split(",") if x.strip()]

for col in ["Tags_Clean", "Industries_Clean", "Malware_Families_Clean", "Attack_IDs_Clean"]:
    if col in otx.columns:
        otx[col] = otx[col].apply(parse_list)

cve["cwes_list"] = cve["cwes_list"].apply(parse_list)
cve["knownRansomwareCampaignUse"] = cve["knownRansomwareCampaignUse"].replace(
    {True: True, False: False, np.nan: False}).astype(bool)
cve["CVE_Year"] = cve["CVE_Year"].astype("Int64")

print("Data ready.")


def save_fig(name):
    path = os.path.join(MODEL_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {name}")


def evaluate_classifier(clf, X_train, y_train, X_test, y_test, name="Model", classes=None):
    """Train and evaluate a classifier, return metrics."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results = {
        "model": name, "accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted
    }

    n_classes = len(np.unique(y_test))
    if y_proba is not None and n_classes == 2:
        results["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
    elif y_proba is not None and n_classes > 2:
        try:
            results["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except Exception:
            results["roc_auc"] = np.nan
    else:
        results["roc_auc"] = np.nan

    return results, y_pred, y_proba


# ============================================================================
# TASK 1: CVE Ransomware Prediction (Binary Classification)
# ============================================================================
print("\n" + "=" * 70)
print("TASK 1: CVE Ransomware Campaign Prediction")
print("=" * 70)

cve["text_combined"] = cve["vulnerabilityName"].fillna("") + " " + cve["shortDescription"].fillna("")
cve["vendor_encoded"] = LabelEncoder().fit_transform(cve["vendorProject"].fillna("missing"))

# Drop rows with no text
cve_model = cve.dropna(subset=["text_combined"]).copy()
cve_model = cve_model[cve_model["text_combined"].str.strip() != ""]

# Feature: TF-IDF on text + year + vendor
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
X_text = tfidf.fit_transform(cve_model["text_combined"])

# Structured features
X_struct = np.column_stack([
    cve_model["CVE_Year"].fillna(0).values,
    cve_model["vendor_encoded"].values,
])
scaler = StandardScaler()
X_struct_scaled = scaler.fit_transform(X_struct)

from scipy.sparse import hstack as sparse_hstack
X_cve = sparse_hstack([X_text, X_struct_scaled])
y_cve = cve_model["knownRansomwareCampaignUse"].values.astype(int)

print(f"CVE Task: {X_cve.shape[0]} samples, {y_cve.sum()} positive ({y_cve.sum()/len(y_cve)*100:.1f}%)")

# Train/test split
X_train_cve, X_test_cve, y_train_cve, y_test_cve = train_test_split(
    X_cve, y_cve, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cve
)

# Define models
cve_models = {}
cve_models["Logistic Regression"] = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
cve_models["Random Forest"] = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE)
cve_models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)

if HAS_XGB:
    scale_pos_weight = (len(y_train_cve) - y_train_cve.sum()) / max(y_train_cve.sum(), 1)
    cve_models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                           scale_pos_weight=scale_pos_weight,
                                           random_state=RANDOM_STATE, verbosity=0)
if HAS_LGB:
    cve_models["LightGBM"] = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                             class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)

# Train & evaluate
cve_results = []
cve_probas = {}
print("\n--- CVE Model Comparison ---")
for name, clf in cve_models.items():
    res, y_pred, y_proba = evaluate_classifier(clf, X_train_cve, y_train_cve, X_test_cve, y_test_cve, name)
    cve_results.append(res)
    cve_probas[name] = y_proba
    print(f"  {name:<25s}  Acc={res['accuracy']:.4f}  F1_macro={res['f1_macro']:.4f}  ROC_AUC={res['roc_auc']:.4f}")

# Best model full report
df_res = pd.DataFrame(cve_results).set_index("model").sort_values("f1_macro", ascending=False)
print(f"\nBest model: {df_res.index[0]}")
best_cve = cve_models[df_res.index[0]]
best_cve.fit(X_train_cve, y_train_cve)
y_pred_cve = best_cve.predict(X_test_cve)
print("\nClassification Report (best model):")
print(classification_report(y_test_cve, y_pred_cve, target_names=["Non-Ransomware", "Ransomware"]))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_cve, X_test_cve, y_test_cve, display_labels=["Non-Ransomware", "Ransomware"],
                                       cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"CVE Ransomware Prediction - {df_res.index[0]}")
save_fig("task1_cve_confusion_matrix.png")

# ROC Curve comparison
fig, ax = plt.subplots(figsize=(8, 6))
for name, clf in cve_models.items():
    if cve_probas[name] is not None and cve_probas[name].ndim == 2 and cve_probas[name].shape[1] >= 2:
        fpr, tpr, _ = roc_curve(y_test_cve, cve_probas[name][:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("CVE Ransomware Prediction - ROC Curves"); ax.legend()
save_fig("task1_cve_roc_curves.png")

# Feature importance (RF)
if hasattr(best_cve, "feature_importances_"):
    importances = best_cve.feature_importances_
elif hasattr(best_cve, "coef_"):
    importances = np.abs(best_cve.coef_[0])
else:
    importances = None

if importances is not None:
    feature_names = list(tfidf.get_feature_names_out())
    feature_names.extend(["CVE_Year", "Vendor_Encoded"])
    indices = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color=sns.color_palette(PALETTE)[0])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis(); ax.set_title("Top 20 Feature Importances (CVE Ransomware)"); ax.set_xlabel("Importance")
    save_fig("task1_cve_feature_importance.png")

df_res.to_csv(os.path.join(MODEL_DIR, "task1_cve_model_comparison.csv"))
print("Task 1 complete.\n")

# ============================================================================
# TASK 2: OTX Industry Classification (Multi-Class)
# ============================================================================
print("\n" + "=" * 70)
print("TASK 2: OTX Primary Industry Classification")
print("=" * 70)

# Map each pulse to its primary industry (first in list), or "Unknown"
def get_primary_industry(ind_list):
    if not ind_list or len(ind_list) == 0:
        return "Unknown"
    return ind_list[0].strip()

otx["Primary_Industry"] = otx["Industries_Clean"].apply(get_primary_industry)

# Keep industries with >= 10 samples
ind_counts = otx["Primary_Industry"].value_counts()
valid_industries = ind_counts[ind_counts >= 10].index.tolist()
otx_ind = otx[otx["Primary_Industry"].isin(valid_industries)].copy()
print(f"Industries kept: {len(valid_industries)} classes, {len(otx_ind)} samples")
print(f"Class distribution:\n{otx_ind['Primary_Industry'].value_counts().to_string()}")

# Text features: combine Tags + malware families as "keywords"
otx_ind["keywords"] = otx_ind.apply(
    lambda r: " ".join(r["Tags_Clean"] + r["Malware_Families_Clean"]), axis=1
)
otx_ind["text_full"] = otx_ind["Title"].fillna("") + " " + otx_ind["keywords"]

# TF-IDF
tfidf_otx = TfidfVectorizer(max_features=800, ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_df=0.8)
X_otx_text = tfidf_otx.fit_transform(otx_ind["text_full"])

# Structured: TLP one-hot
tlp_dummies = pd.get_dummies(otx_ind["TLP"], prefix="TLP").astype(int).values

X_otx = sparse_hstack([X_otx_text, tlp_dummies])
y_otx = LabelEncoder().fit_transform(otx_ind["Primary_Industry"])
class_names_otx = LabelEncoder().fit(otx_ind["Primary_Industry"]).classes_

print(f"OTX Industry Task: {X_otx.shape[0]} samples, {len(class_names_otx)} classes")

X_tr_otx, X_te_otx, y_tr_otx, y_te_otx = train_test_split(
    X_otx, y_otx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_otx
)

otx_models = {}
otx_models["Random Forest"] = RandomForestClassifier(n_estimators=150, max_depth=15, class_weight="balanced",
                                                      random_state=RANDOM_STATE)
if HAS_XGB:
    otx_models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                           random_state=RANDOM_STATE, verbosity=0)
if HAS_LGB:
    otx_models["LightGBM"] = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.1,
                                             class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)

otx_results = []
print("\n--- OTX Industry Model Comparison ---")
for name, clf in otx_models.items():
    res, y_pred, y_proba = evaluate_classifier(clf, X_tr_otx, y_tr_otx, X_te_otx, y_te_otx, name)
    otx_results.append(res)
    print(f"  {name:<25s}  Acc={res['accuracy']:.4f}  F1_macro={res['f1_macro']:.4f}  F1_w={res['f1_weighted']:.4f}")

df_res_otx = pd.DataFrame(otx_results).set_index("model").sort_values("f1_weighted", ascending=False)
print(f"\nBest model: {df_res_otx.index[0]}")

best_otx = otx_models[df_res_otx.index[0]]
best_otx.fit(X_tr_otx, y_tr_otx)
y_pred_otx = best_otx.predict(X_te_otx)
print(f"\nClassification Report:")
print(classification_report(y_te_otx, y_pred_otx, target_names=class_names_otx[:15], zero_division=0))

# Confusion matrix (top 10 classes)
from sklearn.metrics import confusion_matrix as cm_func
cm = cm_func(y_te_otx, y_pred_otx)
top_k = 10
cm_top = cm[:top_k, :top_k]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_top, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_names_otx[:top_k], yticklabels=class_names_otx[:top_k])
ax.set_title(f"OTX Industry Classification - Confusion Matrix (Top {top_k})"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
save_fig("task2_otx_confusion_matrix.png")

# Feature importance
if hasattr(best_otx, "feature_importances_"):
    imp_otx = best_otx.feature_importances_
    fnames = list(tfidf_otx.get_feature_names_out()) + [f"TLP_{c}" for c in ["white","green","amber","red"]]
    idx = np.argsort(imp_otx)[-25:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(idx)), imp_otx[idx], color=sns.color_palette(PALETTE)[1])
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([fnames[i] for i in idx]); ax.invert_yaxis()
    ax.set_title("Top 25 Feature Importances (OTX Industry)"); ax.set_xlabel("Importance")
    save_fig("task2_otx_feature_importance.png")

df_res_otx.to_csv(os.path.join(MODEL_DIR, "task2_otx_model_comparison.csv"))
print("Task 2 complete.\n")

# ============================================================================
# TASK 3: Domain Threat Severity Classification
# ============================================================================
print("\n" + "=" * 70)
print("TASK 3: Domain Threat Severity Classification")
print("=" * 70)

sev_order = ["Low", "Medium", "High"]
domains["Threat_Severity_Num"] = domains["Threat_Severity"].map({s: i for i, s in enumerate(sev_order)})
dom_model = domains.dropna(subset=["Threat_Severity_Num"]).copy()

# Features
dom_features = pd.DataFrame({
    "Domain_Length": dom_model["Domain_Length"].fillna(0),
    "Has_Numbers": dom_model["Has_Numbers"].astype(int).fillna(0),
    "Has_Hyphen": dom_model["Has_Hyphen"].astype(int).fillna(0),
    "Reputation": dom_model["Reputation"].fillna(0),
    "Malicious_Votes": dom_model["Malicious_Votes"].fillna(0),
    "Suspicious_Votes": dom_model["Suspicious_Votes"].fillna(0),
    "Harmless_Votes": dom_model["Harmless_Votes"].fillna(0),
    "Total_Engines": dom_model["Total_Engines"].fillna(0),
    "Popularity_Rank": dom_model["Popularity_Rank"].fillna(0),
})
# Add TLD one-hot (keep top 10, rest as "other")
tld_counts = dom_model["TLD"].value_counts()
top_tlds = tld_counts.head(10).index
dom_model["TLD_grouped"] = dom_model["TLD"].apply(lambda x: x if x in top_tlds else "other")
tld_dummies = pd.get_dummies(dom_model["TLD_grouped"], prefix="TLD").astype(int)
dom_features = pd.concat([dom_features, tld_dummies], axis=1)

X_dom = dom_features.values
y_dom = dom_model["Threat_Severity_Num"].values.astype(int)

print(f"Domain Task: {X_dom.shape[0]} samples, classes={dict(zip(*np.unique(y_dom, return_counts=True)))}")
print(f"  Note: Small dataset ({X_dom.shape[0]} samples) - using 3-fold CV")

# Small dataset: use 3-fold CV
cv_dom = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
X_tr_dom, X_te_dom, y_tr_dom, y_te_dom = train_test_split(
    X_dom, y_dom, test_size=0.3, random_state=RANDOM_STATE, stratify=y_dom
)

dom_models = {}
dom_models["Logistic Regression"] = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
dom_models["Random Forest"] = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced",
                                                      random_state=RANDOM_STATE)
if HAS_XGB:
    dom_models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, verbosity=0)

dom_results = []
print("\n--- Domain Severity Model Comparison ---")
for name, clf in dom_models.items():
    res, y_pred, _ = evaluate_classifier(clf, X_tr_dom, y_tr_dom, X_te_dom, y_te_dom, name)
    dom_results.append(res)
    print(f"  {name:<25s}  Acc={res['accuracy']:.4f}  F1_macro={res['f1_macro']:.4f}")

df_res_dom = pd.DataFrame(dom_results).set_index("model").sort_values("f1_macro", ascending=False)
best_dom = dom_models[df_res_dom.index[0]]
best_dom.fit(X_tr_dom, y_tr_dom)
y_pred_dom = best_dom.predict(X_te_dom)
print(f"\nClassification Report (best: {df_res_dom.index[0]}):")
print(classification_report(y_te_dom, y_pred_dom, target_names=sev_order, zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_dom, X_te_dom, y_te_dom, display_labels=sev_order,
                                       cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"Domain Severity - {df_res_dom.index[0]}")
save_fig("task3_domains_confusion_matrix.png")

df_res_dom.to_csv(os.path.join(MODEL_DIR, "task3_domains_model_comparison.csv"))
print("Task 3 complete.\n")

# ============================================================================
# TASK 4: IP Threat Severity Classification
# ============================================================================
print("\n" + "=" * 70)
print("TASK 4: IP Threat Severity Classification")
print("=" * 70)

ips["Threat_Severity_Num"] = ips["Threat_Severity"].map({s: i for i, s in enumerate(sev_order)})
ips_model = ips.dropna(subset=["Threat_Severity_Num"]).copy()

# Features
ips_features = pd.DataFrame({
    "Malicious_Votes": ips_model["Malicious_Votes"].fillna(0),
    "Suspicious_Votes": ips_model["Suspicious_Votes"].fillna(0),
    "Harmless_Votes": ips_model["Harmless_Votes"].fillna(0),
    "Total_Reports": ips_model["Total_Reports"].fillna(0),
    "Reputation_Score": ips_model["Reputation_Score"].fillna(0),
    "Times_Submitted": ips_model["Times_Submitted"].fillna(0),
    "TOR_Node": ips_model["TOR_Node"].astype(int).fillna(0),
})

# Continent one-hot
cont_dummies = pd.get_dummies(ips_model["Continent"], prefix="Cont").astype(int)
ips_features = pd.concat([ips_features, cont_dummies], axis=1)

# Threat_Label encoding (top 15)
label_counts = ips_model["Threat_Label"].value_counts()
top_labels = label_counts.head(15).index
ips_model["Threat_Label_g"] = ips_model["Threat_Label"].apply(lambda x: x if x in top_labels else "other")
label_dummies = pd.get_dummies(ips_model["Threat_Label_g"], prefix="Label").astype(int)
ips_features = pd.concat([ips_features, label_dummies], axis=1)

X_ips = ips_features.values
y_ips = ips_model["Threat_Severity_Num"].values.astype(int)

print(f"IP Task: {X_ips.shape[0]} samples, classes={dict(zip(*np.unique(y_ips, return_counts=True)))}")

X_tr_ips, X_te_ips, y_tr_ips, y_te_ips = train_test_split(
    X_ips, y_ips, test_size=0.3, random_state=RANDOM_STATE, stratify=y_ips
)

ips_models = {}
ips_models["Logistic Regression"] = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
ips_models["Random Forest"] = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced",
                                                      random_state=RANDOM_STATE)
if HAS_XGB:
    ips_models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE, verbosity=0)
if HAS_LGB:
    ips_models["LightGBM"] = LGBMClassifier(n_estimators=100, max_depth=4, class_weight="balanced",
                                             random_state=RANDOM_STATE, verbose=-1)

ips_results = []
print("\n--- IP Severity Model Comparison ---")
for name, clf in ips_models.items():
    res, y_pred, _ = evaluate_classifier(clf, X_tr_ips, y_tr_ips, X_te_ips, y_te_ips, name)
    ips_results.append(res)
    print(f"  {name:<25s}  Acc={res['accuracy']:.4f}  F1_macro={res['f1_macro']:.4f}")

df_res_ips = pd.DataFrame(ips_results).set_index("model").sort_values("f1_macro", ascending=False)
best_ips = ips_models[df_res_ips.index[0]]
best_ips.fit(X_tr_ips, y_tr_ips)
y_pred_ips = best_ips.predict(X_te_ips)
print(f"\nClassification Report (best: {df_res_ips.index[0]}):")
print(classification_report(y_te_ips, y_pred_ips, target_names=sev_order, zero_division=0))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_ips, X_te_ips, y_te_ips, display_labels=sev_order,
                                       cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"IP Severity - {df_res_ips.index[0]}")
save_fig("task4_ips_confusion_matrix.png")

# Feature importance
if hasattr(best_ips, "feature_importances_"):
    imp_ips = best_ips.feature_importances_
    fnames_ips = list(ips_features.columns)
    idx = np.argsort(imp_ips)[-15:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(idx)), imp_ips[idx], color=sns.color_palette(PALETTE)[3])
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([fnames_ips[i] for i in idx]); ax.invert_yaxis()
    ax.set_title("Top 15 Feature Importances (IP Severity)"); ax.set_xlabel("Importance")
    save_fig("task4_ips_feature_importance.png")

df_res_ips.to_csv(os.path.join(MODEL_DIR, "task4_ips_model_comparison.csv"))
print("Task 4 complete.\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE ENGINEERING & MODELING - SUMMARY")
print("=" * 70)

summary_rows = [
    ("Task 1: CVE Ransomware", f"{X_cve.shape[0]} samples, {y_cve.sum()} positive", df_res.index[0], f"{df_res.iloc[0]['f1_macro']:.4f}"),
    ("Task 2: OTX Industry", f"{X_otx.shape[0]} samples, {len(class_names_otx)} classes", df_res_otx.index[0], f"{df_res_otx.iloc[0]['f1_weighted']:.4f}"),
    ("Task 3: Domain Severity", f"{X_dom.shape[0]} samples, 3 classes", df_res_dom.index[0], f"{df_res_dom.iloc[0]['f1_macro']:.4f}"),
    ("Task 4: IP Severity", f"{X_ips.shape[0]} samples, 3 classes", df_res_ips.index[0], f"{df_res_ips.iloc[0]['f1_macro']:.4f}"),
]

print(f"\n{'Task':<30} {'Data':<30} {'Best Model':<25} {'F1'}");
print("-" * 110)
for row in summary_rows:
    print(f"{row[0]:<30} {row[1]:<30} {row[2]:<25} {row[3]:>6}")

print(f"\nAll outputs saved to: {MODEL_DIR}")
print(f"  - Model comparison CSV files (4)")
print(f"  - Confusion matrix plots (4)")
print(f"  - ROC curves (Task 1)")
print(f"  - Feature importance plots (Task 1, 2, 4)")
print(f"\nScripts: explore_and_clean.py | eda_analysis.py | feature_modeling.py")
