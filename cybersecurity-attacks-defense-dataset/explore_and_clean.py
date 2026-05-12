# -*- coding: utf-8 -*-
"""
Data Exploration & Cleaning Script - Cybersecurity Datasets
4 CSV files: OTX Threat Intel, CVE Vulnerabilities, Malicious Domains, Malicious IPs
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

BASE_DIR = r"E:\py_project\Cybersecurity Attacks & Defense Dataset 2026"
OUT_DIR = os.path.join(BASE_DIR, "cleaned")
os.makedirs(OUT_DIR, exist_ok=True)

UNKNOWN_REPLACE = ["Unknown", "unknown", "UNKNOWN", ""]


def load_csv_robust(filepath, **kwargs):
    """Load CSV with multi-line field support (e.g. WHOIS summaries)"""
    try:
        df = pd.read_csv(filepath, **kwargs)
    except Exception:
        df = pd.read_csv(filepath, engine="python", on_bad_lines="warn", **kwargs)
    return df


def basic_info(df, name):
    """Print basic dataset info"""
    print(f"\n{'='*70}")
    print(f"[INFO] {name}")
    print(f"{'='*70}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count}")


def replace_unknowns(df):
    """Replace 'Unknown', 'unknown', '' with NaN and strip whitespace"""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
            df[col] = df[col].replace(UNKNOWN_REPLACE, np.nan)
    return df


def parse_dates_safe(series):
    """Try multiple date formats"""
    return pd.to_datetime(series, errors="coerce")


# ============================================================================
# 1. OTX Threat Intelligence
# ============================================================================
print("\n" + "#" * 70)
print("# 1. OTX Threat Intelligence")
print("#" * 70)

otx = load_csv_robust(os.path.join(BASE_DIR, "1_otx_threat_intel.csv"))
basic_info(otx, "OTX Raw")

# --- Cleaning ---
otx = replace_unknowns(otx)

# Parse dates
otx["Created"] = parse_dates_safe(otx["Created"])
otx["Modified"] = parse_dates_safe(otx["Modified"])

# Numeric fields
otx["Indicators_Count"] = pd.to_numeric(otx["Indicators_Count"], errors="coerce")
otx["Subscribers"] = pd.to_numeric(otx["Subscribers"], errors="coerce")

# Split multi-value fields (Tags, Attack_IDs, Industries, Countries, Malware_Families)
for col in ["Tags", "Attack_IDs", "Industries", "Countries", "Malware_Families"]:
    if col in otx.columns:
        otx[f"{col}_Clean"] = otx[col].apply(
            lambda x: [i.strip() for i in str(x).split(",") if i.strip() and i.strip() not in UNKNOWN_REPLACE]
            if isinstance(x, str) else []
        )

# Deduplicate by Pulse_ID
otx = otx.drop_duplicates(subset="Pulse_ID").reset_index(drop=True)

print(f"\nOTX Cleaned Shape: {otx.shape}")

# ============================================================================
# 2. CVE Vulnerabilities
# ============================================================================
print("\n" + "#" * 70)
print("# 2. CVE Vulnerabilities")
print("#" * 70)

cve = load_csv_robust(os.path.join(BASE_DIR, "2_cve_vulnerabilities.csv"))
basic_info(cve, "CVE Raw")

# --- Cleaning ---
cve = replace_unknowns(cve)

cve["dateAdded"] = parse_dates_safe(cve["dateAdded"])
cve["dueDate"] = parse_dates_safe(cve["dueDate"])

# Split CWEs
def clean_cwes(val):
    if isinstance(val, str):
        parts = re.split(r",\s*", val.strip('"'))
        return [p.strip() for p in parts if p.strip()]
    return []

cve["cwes_list"] = cve["cwes"].apply(clean_cwes)

# Extract CVE year
cve["CVE_Year"] = cve["cveID"].str.extract(r"CVE-(\d{4})").astype("Int64")

# knownRansomwareCampaignUse -> boolean
cve["knownRansomwareCampaignUse"] = cve["knownRansomwareCampaignUse"].map(
    {"Known": True}
).astype("boolean")

cve = cve.drop_duplicates(subset="cveID").reset_index(drop=True)

print(f"\nCVE Cleaned Shape: {cve.shape}")

# ============================================================================
# 3. Malicious Domains
# ============================================================================
print("\n" + "#" * 70)
print("# 3. Malicious Domains")
print("#" * 70)

domains = load_csv_robust(os.path.join(BASE_DIR, "3_malicious_domains.csv"))
basic_info(domains, "Domains Raw")

# --- Cleaning ---
domains = replace_unknowns(domains)

# Parse Unix timestamps
domains["Creation_Date"] = pd.to_numeric(domains["Creation_Date"], errors="coerce")
domains["Last_Update_Date"] = pd.to_numeric(domains["Last_Update_Date"], errors="coerce")
domains["Last_Analysis_Date"] = pd.to_numeric(domains["Last_Analysis_Date"], errors="coerce")
domains["Popularity_Rank"] = pd.to_numeric(domains["Popularity_Rank"], errors="coerce")

# Numeric columns
numeric_cols = ["Domain_Length", "Malicious_Votes", "Suspicious_Votes", "Harmless_Votes",
                "Undetected_Votes", "Total_Engines", "Reputation"]
for c in numeric_cols:
    domains[c] = pd.to_numeric(domains[c], errors="coerce")

# Has_Numbers / Has_Hyphen -> bool
domains["Has_Numbers"] = domains["Has_Numbers"].map({"Yes": True, "No": False}).astype("boolean")
domains["Has_Hyphen"] = domains["Has_Hyphen"].map({"Yes": True, "No": False}).astype("boolean")

domains = domains.drop_duplicates().reset_index(drop=True)

print(f"\nDomains Cleaned Shape: {domains.shape}")

# ============================================================================
# 4. Malicious IPs
# ============================================================================
print("\n" + "#" * 70)
print("# 4. Malicious IPs")
print("#" * 70)

ips = load_csv_robust(os.path.join(BASE_DIR, "4_malicious_ips.csv"))
basic_info(ips, "IPs Raw")

# --- Cleaning ---
ips = replace_unknowns(ips)

# Numeric fields
ips["ASN"] = pd.to_numeric(ips["ASN"], errors="coerce").astype("Int64")
for c in ["Malicious_Votes", "Suspicious_Votes", "Harmless_Votes", "Undetected_Votes",
          "Total_Reports", "Reputation_Score", "Times_Submitted"]:
    ips[c] = pd.to_numeric(ips[c], errors="coerce")

ips["Last_Analysis_Date"] = pd.to_numeric(ips["Last_Analysis_Date"], errors="coerce")

# TOR_Node -> bool
ips["TOR_Node"] = ips["TOR_Node"].map({"Yes": True, "No": False}).astype("boolean")

ips = ips.drop_duplicates(subset="IP").reset_index(drop=True)

print(f"\nIPs Cleaned Shape: {ips.shape}")

# ============================================================================
# Save Cleaned Data
# ============================================================================
print("\n" + "#" * 70)
print("# Saving Cleaned Data")
print("#" * 70)

otx.to_csv(os.path.join(OUT_DIR, "1_otx_threat_intel_clean.csv"), index=False)
cve.to_csv(os.path.join(OUT_DIR, "2_cve_vulnerabilities_clean.csv"), index=False)
domains.to_csv(os.path.join(OUT_DIR, "3_malicious_domains_clean.csv"), index=False)
ips.to_csv(os.path.join(OUT_DIR, "4_malicious_ips_clean.csv"), index=False)

print(f"Cleaned files saved to: {OUT_DIR}")
for f in os.listdir(OUT_DIR):
    fpath = os.path.join(OUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"   {f}  ({size_kb:.1f} KB)")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "#" * 70)
print("# Summary Report")
print("#" * 70)

print(f"\n{'Dataset':<30} {'Records':>8} {'Columns':>8} {'Missing%':>10}")
print("-" * 60)
for name, df in [("OTX Threat Intel", otx), ("CVE Vulnerabilities", cve),
                  ("Malicious Domains", domains), ("Malicious IPs", ips)]:
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    pct = missing_cells / total_cells * 100 if total_cells > 0 else 0
    print(f"{name:<30} {df.shape[0]:>8} {df.shape[1]:>8} {pct:>9.2f}%")

# OTX statistics
print(f"\n--- OTX: Top 10 Malware Families ---")
all_families = []
for families in otx["Malware_Families_Clean"]:
    all_families.extend(families)
family_counts = pd.Series(all_families).value_counts().head(10)
if len(family_counts) > 0:
    print(family_counts.to_string())
else:
    print("(no data)")

print(f"\n--- OTX: Top 10 Targeted Industries ---")
all_industries = []
for ind in otx["Industries_Clean"]:
    all_industries.extend(ind)
ind_counts = pd.Series(all_industries).value_counts().head(10)
if len(ind_counts) > 0:
    print(ind_counts.to_string())
else:
    print("(no data)")

# CVE statistics
print(f"\n--- CVE: Year Distribution ---")
print(cve["CVE_Year"].value_counts().sort_index().to_string())

print(f"\n--- CVE: Ransomware Campaign Use ---")
print(cve["knownRansomwareCampaignUse"].value_counts(dropna=False).to_string())

# Domain & IP statistics
print(f"\n--- Domains: Threat Severity Distribution ---")
print(domains["Threat_Severity"].value_counts().to_string())

print(f"\n--- IPs: Threat Severity Distribution ---")
print(ips["Threat_Severity"].value_counts().to_string())

print("\n" + "=" * 70)
print("Data Exploration & Cleaning Complete!")
print("=" * 70)
