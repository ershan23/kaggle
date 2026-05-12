# Cybersecurity Attacks & Defense Dataset 2026

A comprehensive pipeline for cybersecurity threat data exploration, cleaning, visualization, feature engineering, and machine learning modeling. Designed for SOC analysts, threat intelligence researchers, and security data scientists.

## Dataset Overview

| Dataset | Source | Records | Description |
|---------|--------|---------|-------------|
| `1_otx_threat_intel.csv` | AlienVault OTX | 2,365 | Global threat intelligence pulses: malware families, ATT&CK techniques, targeted industries/countries |
| `2_cve_vulnerabilities.csv` | CISA KEV + NVD | 1,585 | Actively exploited vulnerabilities (2024-2026) with CVSS, vendor info, ransomware use, and patch deadlines |
| `3_malicious_domains.csv` | VirusTotal | 162 | Domains flagged for phishing, C2, malware distribution with registrar, reputation, and WHOIS data |
| `4_malicious_ips.csv` | VirusTotal | 200 | IPs used for scanning, brute-force, botnet activity with ASN, geolocation, and TOR node indicators |

## Project Structure

```
.
├── 1_otx_threat_intel.csv              # Raw OTX threat intelligence
├── 2_cve_vulnerabilities.csv           # Raw CVE vulnerabilities
├── 3_malicious_domains.csv             # Raw malicious domains
├── 4_malicious_ips.csv                 # Raw malicious IPs
├── explore_and_clean.py                # Step 1: Data cleaning & initial exploration
├── eda_analysis.py                     # Step 2: EDA with 32 statistical charts
├── feature_modeling.py                 # Step 3: Feature engineering + 4 ML classification tasks
├── dashboard.py                        # Step 4: Interactive Streamlit dashboard
├── cleaned/                            # Output: cleaned CSV files
├── eda_charts/                         # Output: 32 EDA visualization PNGs
├── model_outputs/                      # Output: model metrics, confusion matrices, feature importance
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Conda environment at `E:\miniconda3\envs\pytorch`

```powershell
# Activate environment
conda activate pytorch

# Install dependencies (if needed)
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm streamlit
```

### Run the Pipeline

```powershell
# Step 1: Data Cleaning
python explore_and_clean.py

# Step 2: Exploratory Data Analysis (32 charts)
python eda_analysis.py

# Step 3: Feature Engineering & Modeling (4 classifiers)
python feature_modeling.py

# Step 4: Launch Interactive Dashboard
streamlit run dashboard.py
```

## Step-by-Step Details

### 1. Data Cleaning (`explore_and_clean.py`)

| Operation | Details |
|-----------|---------|
| Null handling | `"Unknown"`, `""`, and variants → `NaN` |
| Type correction | Dates parsed to `datetime64`, numeric strings to `float/int`, booleans converted |
| Multi-value splitting | Tags, Attack_IDs, Industries, Countries → clean Python lists |
| Deduplication | 25 duplicate Pulse_IDs removed from OTX (2,365 → 2,340) |
| Feature extraction | CVE year extracted from CVE ID, CWE codes split to list |
| Output | 4 cleaned CSV files in `cleaned/` directory |

### 2. Exploratory Data Analysis (`eda_analysis.py`)

Generates **32 charts** across 5 groups:

| Group | Charts | Key Insights |
|-------|--------|--------------|
| OTX (8) | Temporal trend, TLP distribution, Top malware/ATT&CK/industries/countries/tags, scatter plots | Cobalt Strike (#1), T1027 Obfuscation dominates (996 pulses) |
| CVE (7) | Year distribution + ransomware overlay, top vendors/products/CWEs, response window, monthly trend | 317 CVEs (20%) known ransomware use, median patch window ~14 days |
| Domains (6) | Severity pie, TLD distribution, length/features vs severity, registrar analysis | 89.5% Low severity, `.com` most common TLD |
| IPs (9) | Severity pie, country/continent, reputation, threat labels/categories, TOR analysis, ownership | EU hosts 48% of malicious IPs, 3% TOR exit nodes |
| Cross (2) | Summary dashboard, correlation heatmaps | Numeric feature correlations for domains and IPs |

### 3. Feature Engineering & Modeling (`feature_modeling.py`)

Four supervised classification tasks with model comparison:

| Task | Samples | Type | Classes | Best Model | F1 |
|------|---------|------|---------|-------------|-----|
| CVE Ransomware Prediction | 1,585 | Binary | 2 (20% positive) | Random Forest | 0.67 macro |
| OTX Industry Classification | 2,256 | Multi-class | 15 (54% Unknown) | XGBoost | 0.56 weighted |
| Domain Threat Severity | 162 | Multi-class | 3 (89% Low) | XGBoost | 1.00 macro* |
| IP Threat Severity | 200 | Multi-class | 3 (87% Low) | XGBoost | 1.00 macro* |

> **\*Note**: Domain and IP tasks have small sample sizes (162/200) with heavy class imbalance. Perfect scores indicate potential overfitting; production deployment requires more data.

**Features engineered:**
- Text: TF-IDF vectorization (500-800 features, 1-2 ngrams, sublinear scaling)
- Structured: vendor encoding, CVE year, TLP one-hot, domain length, reputation scores, TOR node flag
- Combined: sparse feature stacking for efficient pipeline processing

**Models compared:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM

**Outputs in `model_outputs/`:**
- 4 model comparison CSVs
- 4 confusion matrix plots
- ROC curve comparison (CVE task)
- 3 feature importance charts

### 4. Interactive Dashboard (`dashboard.py`)

Streamlit dashboard with **7 pages**:

| Page | Content |
|------|---------|
| **Overview** | KPI cards, dataset distribution, monthly trends for OTX and CVE |
| **OTX Threat Intel** | Malware families, ATT&CK techniques, targeted industries/countries, TLP distribution, daily burst detection |
| **CVE Vulnerabilities** | Year distribution with ransomware overlay, top vendors/CWEs, response window analysis |
| **Malicious Domains** | Severity pie, TLDs, domain features (numbers/hyphens), high-severity table |
| **Malicious IPs** | Geographic distribution, threat categories, reputation scores, TOR analysis, high-severity table |
| **Model Results** | Task-level metrics, model comparison tables, performance summaries |
| **Recommendations** | 3-priority actionable recommendations with KPI summary matrix |

## Key Findings at a Glance

### Threat Landscape
- **Top Malware**: Cobalt Strike (62), Lumma Stealer (46), AsyncRAT (43), XMRig (35)
- **Top ATT&CK**: T1027 Obfuscation (996), T1082 System Discovery (749), T1059 Scripting (618)
- **Top Industries**: Government (547), Finance (395), Technology (343)
- **Top Countries**: United States, India, United Kingdom, Germany, Brazil

### Vulnerability Management
- 1,585 actively exploited CVEs, 317 (20%) used in ransomware campaigns
- **Top Vendors**: Microsoft, Cisco, Apple, Google, Apache
- **Top CWEs**: CWE-416, CWE-787, CWE-22, CWE-119, CWE-79
- Median patch window: **14 days** (aligns with BOD 22-01 mandate)
- Vulnerabilities span **2002-2026**, peak at 213 in 2021

### Malicious Infrastructure
- Domain severity: Low 89.5%, Medium 6.2%, High 4.3%
- IP severity: Low 87%, Medium 7%, High 6%
- 3% of IPs are TOR exit nodes
- Geographic concentration: Europe (48.4%), Asia (32.3%)

## Recommendations

### Priority 1 - Immediate Actions
1. **Patch ransomware-exploitable CVEs** within 14 days (317 CVEs with known ransomware use)
2. **Deploy AMSI bypass protection, PowerShell logging, ScriptBlock monitoring** to counter T1027/T1059 dominance
3. **Block high-severity domains/IPs** in SIEM/firewall immediately

### Priority 2 - Strategic Improvements
1. **Build industry-specific defense playbooks** (Government, Finance, Technology are top 3 targets)
2. **Automate domain/IP threat feeds** into security infrastructure daily
3. **Monitor `.top`, `.xyz`, `.tk` TLD registrations** for higher malicious probability

### Priority 3 - Long-term Investments
1. **Deploy CVE Ransomware Classifier** (Random Forest, F1=0.67) as an automated triage tool
2. **Use NER** (spaCy/transformers) to extract industry entities from 54% of unlabeled threat descriptions
3. **Collect more labeled data** for domain/IP models (target: 1,000+ each)
4. **Integrate additional enrichment sources**: CVSS vectors, EPSS scores, passive DNS, SSL certificates

## Tech Stack

| Component | Libraries |
|-----------|-----------|
| Data Processing | pandas, numpy, scipy |
| Visualization | matplotlib, seaborn |
| ML Modeling | scikit-learn, XGBoost, LightGBM |
| Dashboard | Streamlit |
| Environment | Python 3.9+, Miniconda (pytorch env) |

## License

This project is for educational and research purposes. Dataset sources are publicly available from AlienVault OTX, CISA KEV, NVD, and VirusTotal.
