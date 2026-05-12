# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis - Cybersecurity Datasets
Visualizes distributions, trends, correlations across all 4 datasets.
Outputs charts to /eda_charts/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import re
import ast
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = r"E:\py_project\Cybersecurity Attacks & Defense Dataset 2026"
CLEAN_DIR = os.path.join(BASE_DIR, "cleaned")
CHART_DIR = os.path.join(BASE_DIR, "eda_charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.figsize": (12, 6),
    "savefig.bbox": "tight",
})
sns.set_style("whitegrid")

# Color palettes
PALETTE = "Set2"
SEQUENTIAL = "YlOrRd"

# ============================================================================
# Load Data
# ============================================================================
print("Loading cleaned datasets...")

otx = pd.read_csv(os.path.join(CLEAN_DIR, "1_otx_threat_intel_clean.csv"))
cve = pd.read_csv(os.path.join(CLEAN_DIR, "2_cve_vulnerabilities_clean.csv"))
domains = pd.read_csv(os.path.join(CLEAN_DIR, "3_malicious_domains_clean.csv"))
ips = pd.read_csv(os.path.join(CLEAN_DIR, "4_malicious_ips_clean.csv"))

# Fix date columns
otx["Created"] = pd.to_datetime(otx["Created"], errors="coerce")
otx["Modified"] = pd.to_datetime(otx["Modified"], errors="coerce")
cve["dateAdded"] = pd.to_datetime(cve["dateAdded"], errors="coerce")
cve["dueDate"] = pd.to_datetime(cve["dueDate"], errors="coerce")

# Fix list columns (loaded as stringified lists from CSV)
def parse_list_column(series):
    def _parse(val):
        if pd.isna(val) or val == "[]" or val == "":
            return []
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return [x.strip() for x in str(val).split(",") if x.strip()]
    return series.apply(_parse)

for col in ["Tags_Clean", "Attack_IDs_Clean", "Industries_Clean", "Countries_Clean", "Malware_Families_Clean"]:
    if col in otx.columns:
        otx[col] = parse_list_column(otx[col])

cve["cwes_list"] = parse_list_column(cve["cwes_list"])
cve["knownRansomwareCampaignUse"] = cve["knownRansomwareCampaignUse"].replace({True: True, np.nan: False}).astype(bool)
cve["CVE_Year"] = cve["CVE_Year"].astype("Int64")

# Dates as timestamps -> readable
for df in [domains, ips]:
    for col in df.columns:
        if "Date" in col and df[col].dtype in [np.float64, np.int64]:
            mask = df[col] > 1e10  # likely Unix timestamp
            df.loc[mask, col] = pd.to_datetime(df.loc[mask, col], unit="s", errors="coerce")
            df[col] = pd.to_datetime(df[col], errors="coerce")

# Fix IPS ASN
ips["ASN"] = ips["ASN"].astype("Int64")

# Threat severity order
severity_order = ["Low", "Medium", "High", "Critical"]

print("Data loaded & preprocessed.")

# ============================================================================
# Helper functions
# ============================================================================
def safe_save(name):
    path = os.path.join(CHART_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {name}")

def count_label(ax, fontsize=9):
    for bar in ax.containers:
        ax.bar_label(bar, fontsize=fontsize, padding=2)

# ============================================================================
# 1. OTX THREAT INTELLIGENCE
# ============================================================================
print("\n" + "=" * 60)
print("1. OTX THREAT INTELLIGENCE EDA")
print("=" * 60)

# 1.1 Temporal trend: pulses per month
fig, ax = plt.subplots(figsize=(14, 5))
otx_monthly = otx.set_index("Created").resample("ME").size()
ax.fill_between(otx_monthly.index, otx_monthly.values, alpha=0.4, color=sns.color_palette(PALETTE)[0])
ax.plot(otx_monthly.index, otx_monthly.values, marker="o", color=sns.color_palette(PALETTE)[0])
ax.set_title("OTX Threat Pulses Over Time (Monthly)")
ax.set_xlabel("Date"); ax.set_ylabel("Number of Pulses")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: pd.Timestamp(matplotlib.dates.num2date(x)).strftime("%Y-%m")))
safe_save("01_otx_temporal_trend.png")

# 1.2 TLP Distribution
fig, ax = plt.subplots(figsize=(8, 5))
tlp_counts = otx["TLP"].value_counts()
colors = {"white": "#81C784", "green": "#66BB6A", "amber": "#FFB74D", "red": "#E57373"}
bar_colors = [colors.get(k, "#BDBDBD") for k in tlp_counts.index]
bars = ax.bar(tlp_counts.index, tlp_counts.values, color=bar_colors, edgecolor="white")
count_label(ax)
ax.set_title("OTX TLP Distribution"); ax.set_ylabel("Count")
safe_save("02_otx_tlp_distribution.png")

# 1.3 Top 15 Malware Families
all_malware = []
for families in otx["Malware_Families_Clean"]:
    all_malware.extend([f.strip() for f in families if f.strip()])
malware_counts = pd.Series(all_malware).value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(malware_counts)), malware_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(malware_counts)))
ax.set_yticklabels(malware_counts.index)
ax.invert_yaxis()
ax.set_title("Top 15 Malware Families in OTX"); ax.set_xlabel("Count")
for i, v in enumerate(malware_counts.values):
    ax.text(v + 1, i, str(v), va="center", fontsize=9)
safe_save("03_otx_top_malware.png")

# 1.4 Top 20 ATT&CK Techniques
all_attacks = []
for atk in otx["Attack_IDs_Clean"]:
    all_attacks.extend([a.strip() for a in atk if a.strip()])
attack_counts = pd.Series(all_attacks).value_counts().head(20)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(range(len(attack_counts)), attack_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(attack_counts)))
ax.set_yticklabels(attack_counts.index)
ax.invert_yaxis()
ax.set_title("Top 20 ATT&CK Techniques"); ax.set_xlabel("Count")
for i, v in enumerate(attack_counts.values):
    ax.text(v + 1, i, str(v), va="center", fontsize=8)
safe_save("04_otx_top_attack_techniques.png")

# 1.5 Top Industries targeted
all_ind = []
for ind in otx["Industries_Clean"]:
    all_ind.extend([i.strip() for i in ind if i.strip()])
ind_counts = pd.Series(all_ind).value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(ind_counts)), ind_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(ind_counts)))
ax.set_yticklabels(ind_counts.index)
ax.invert_yaxis()
ax.set_title("Top 15 Targeted Industries"); ax.set_xlabel("Count")
count_label(ax)
safe_save("05_otx_top_industries.png")

# 1.6 Top Countries targeted
all_ctry = []
for c in otx["Countries_Clean"]:
    all_ctry.extend([x.strip() for x in c if x.strip()])
ctry_counts = pd.Series(all_ctry).value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(ctry_counts)), ctry_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(ctry_counts)))
ax.set_yticklabels(ctry_counts.index)
ax.invert_yaxis()
ax.set_title("Top 15 Targeted Countries"); ax.set_xlabel("Count")
count_label(ax)
safe_save("06_otx_top_countries.png")

# 1.7 Indicators_Count vs Subscribers scatter
fig, ax = plt.subplots(figsize=(8, 6))
mask = (otx["Indicators_Count"] > 0) & (otx["Subscribers"] > 0)
n_pos = mask.sum()
if n_pos > 0:
    ax.scatter(otx.loc[mask, "Indicators_Count"], otx.loc[mask, "Subscribers"],
               alpha=0.4, c=sns.color_palette(PALETTE)[0], edgecolors="none")
    ax.set_title(f"Indicators Count vs Subscribers (n={n_pos} with data)")
else:
    ax.text(0.5, 0.5, "No positive values for Indicators or Subscribers", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Indicators Count vs Subscribers (no data)")
ax.set_xlabel("Indicators Count"); ax.set_ylabel("Subscribers")
safe_save("07_otx_indicators_vs_subscribers.png")

# 1.8 Top Tags word frequency
all_tags = []
for t in otx["Tags_Clean"]:
    all_tags.extend([x.strip() for x in t if x.strip()])
tag_counts = pd.Series(all_tags).value_counts().head(25)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(tag_counts)), tag_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(tag_counts)))
ax.set_yticklabels(tag_counts.index)
ax.invert_yaxis()
ax.set_title("Top 25 Threat Tags"); ax.set_xlabel("Count")
count_label(ax)
safe_save("08_otx_top_tags.png")

print("OTX EDA complete (8 charts).")

# ============================================================================
# 2. CVE VULNERABILITIES
# ============================================================================
print("\n" + "=" * 60)
print("2. CVE VULNERABILITIES EDA")
print("=" * 60)

# 2.1 CVE Year distribution with ransomware overlay
yearly = cve.groupby("CVE_Year").agg(total=("cveID", "count"), ransomware=("knownRansomwareCampaignUse", "sum"))
yearly = yearly[yearly.index.notna()]

fig, ax = plt.subplots(figsize=(14, 6))
x = yearly.index.astype(int)
ax.bar(x, yearly["total"], label="All CVEs", color=sns.color_palette(PALETTE)[0], alpha=0.8)
ax.bar(x, yearly["ransomware"], label="Ransomware-Known", color="#E57373", alpha=0.9)
ax.set_title("CVE Vulnerabilities by Year (with Ransomware Campaign Use)")
ax.set_xlabel("Year"); ax.set_ylabel("Count")
ax.legend()
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=15))
safe_save("09_cve_year_distribution.png")

# 2.2 Top Vendors
top_vendors = cve["vendorProject"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(top_vendors)), top_vendors.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(top_vendors)))
ax.set_yticklabels(top_vendors.index)
ax.invert_yaxis()
ax.set_title("Top 15 Vendors with Most CVEs"); ax.set_xlabel("Count")
count_label(ax)
safe_save("10_cve_top_vendors.png")

# 2.3 Days between dateAdded and dueDate (response urgency)
cve["response_days"] = (cve["dueDate"] - cve["dateAdded"]).dt.days
cve_r = cve[cve["response_days"].notna() & (cve["response_days"] >= 0) & (cve["response_days"] < 60)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(cve_r["response_days"], bins=30, color=sns.color_palette(PALETTE)[0], edgecolor="white", alpha=0.8)
ax.axvline(cve_r["response_days"].median(), color="#E57373", linestyle="--", linewidth=2, label=f"Median: {cve_r['response_days'].median():.0f} days")
ax.axvline(14, color="#FFB74D", linestyle="--", linewidth=1.5, label="BOD 22-01: 14 days")
ax.set_title("CVE Response Window (dateAdded -> dueDate)")
ax.set_xlabel("Days"); ax.set_ylabel("Count")
ax.legend()
safe_save("11_cve_response_window.png")

# 2.4 Top CWEs
all_cwes = []
for cw in cve["cwes_list"]:
    all_cwes.extend([x.strip() for x in cw if x.strip()])
cwe_counts = pd.Series(all_cwes).value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(cwe_counts)), cwe_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(cwe_counts)))
ax.set_yticklabels(cwe_counts.index)
ax.invert_yaxis()
ax.set_title("Top 15 CWE Weaknesses"); ax.set_xlabel("Count")
count_label(ax)
safe_save("12_cve_top_cwes.png")

# 2.5 Ransomware Campaign Use breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie
ransom_counts = cve["knownRansomwareCampaignUse"].value_counts()
axes[0].pie(ransom_counts.values, labels=["Not Known", "Known Ransomware"], autopct="%1.1f%%",
            colors=[sns.color_palette(PALETTE)[0], "#E57373"], startangle=90, explode=(0, 0.05))
axes[0].set_title("Ransomware Campaign Usage")

# By year (heat data)
ransom_by_year = cve.groupby("CVE_Year")["knownRansomwareCampaignUse"].agg(["count", "sum"])
ransom_by_year["pct"] = ransom_by_year["sum"] / ransom_by_year["count"] * 100
ransom_by_year = ransom_by_year[ransom_by_year.index.notna()]
axes[1].bar(ransom_by_year.index.astype(int), ransom_by_year["pct"], color="#E57373", alpha=0.8)
axes[1].set_title("Ransomware CVEs % by Year")
axes[1].set_xlabel("Year"); axes[1].set_ylabel("% Ransomware")
safe_save("13_cve_ransomware_analysis.png")

# 2.6 Monthly CVE additions
cve_monthly = cve.set_index("dateAdded").resample("ME").size()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(cve_monthly.index, cve_monthly.values, alpha=0.4, color=sns.color_palette(PALETTE)[2])
ax.plot(cve_monthly.index, cve_monthly.values, marker="o", markersize=3, color=sns.color_palette(PALETTE)[2])
ax.set_title("CVE Additions Over Time (Monthly)")
ax.set_xlabel("Date"); ax.set_ylabel("Count")
safe_save("14_cve_monthly_trend.png")

# 2.7 Top Products
top_products = cve["product"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(top_products)), top_products.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(top_products)))
ax.set_yticklabels(top_products.index)
ax.invert_yaxis()
ax.set_title("Top 15 Products with Most CVEs"); ax.set_xlabel("Count")
count_label(ax)
safe_save("15_cve_top_products.png")

print("CVE EDA complete (7 charts).")

# ============================================================================
# 3. MALICIOUS DOMAINS
# ============================================================================
print("\n" + "=" * 60)
print("3. MALICIOUS DOMAINS EDA")
print("=" * 60)

# 3.1 Threat Severity Distribution
fig, ax = plt.subplots(figsize=(7, 5))
sev = domains["Threat_Severity"].value_counts()
order = [s for s in severity_order if s in sev.index]
ax.pie(sev[order].values, labels=order, autopct="%1.1f%%",
       colors=sns.color_palette(PALETTE), startangle=90)
ax.set_title("Domain Threat Severity Distribution")
safe_save("16_domains_severity.png")

# 3.2 Top TLDs
tld_counts = domains["TLD"].value_counts().head(12)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(tld_counts.index, tld_counts.values, color=sns.color_palette(PALETTE), edgecolor="white")
count_label(ax); ax.set_title("Top 12 TLDs in Malicious Domains"); ax.set_ylabel("Count")
safe_save("17_domains_top_tlds.png")

# 3.3 Domain Length distribution by severity
fig, ax = plt.subplots(figsize=(10, 5))
for i, sev in enumerate([s for s in severity_order if s in domains["Threat_Severity"].values]):
    subset = domains[domains["Threat_Severity"] == sev]["Domain_Length"]
    ax.hist(subset, bins=20, alpha=0.5, label=f"{sev} (n={len(subset)})",
            color=sns.color_palette(PALETTE)[i])
ax.set_title("Domain Length Distribution by Threat Severity")
ax.set_xlabel("Length"); ax.set_ylabel("Count"); ax.legend()
safe_save("18_domains_length_by_severity.png")

# 3.4 Has_Numbers / Has_Hyphen vs Severity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ct_nums = pd.crosstab(domains["Threat_Severity"], domains["Has_Numbers"], normalize="index") * 100
ct_nums.plot(kind="bar", ax=axes[0], color=sns.color_palette(PALETTE)[:2], edgecolor="white")
axes[0].set_title("Has Numbers vs Severity (%)"); axes[0].set_ylabel("%"); axes[0].legend(title="Has Numbers")

ct_hyp = pd.crosstab(domains["Threat_Severity"], domains["Has_Hyphen"], normalize="index") * 100
ct_hyp.plot(kind="bar", ax=axes[1], color=sns.color_palette(PALETTE)[:2], edgecolor="white")
axes[1].set_title("Has Hyphen vs Severity (%)"); axes[1].set_ylabel("%"); axes[1].legend(title="Has Hyphen")
safe_save("19_domains_features_vs_severity.png")

# 3.5 Reputation vs Malicious Votes
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(domains["Reputation"], domains["Malicious_Votes"],
                c=domains["Threat_Severity"].map({"Low": 0, "Medium": 1, "High": 2, "Critical": 3}),
                cmap=SEQUENTIAL, alpha=0.7, edgecolors="none")
ax.set_title("Reputation Score vs Malicious Votes"); ax.set_xlabel("Reputation"); ax.set_ylabel("Malicious Votes")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Severity (0=Low, 3=Critical)")
safe_save("20_domains_reputation_vs_votes.png")

# 3.6 Top Registrars
reg_counts = domains["Registrar"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(reg_counts)), reg_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(reg_counts)))
ax.set_yticklabels(reg_counts.index)
ax.invert_yaxis()
ax.set_title("Top 10 Registrars"); ax.set_xlabel("Count")
count_label(ax)
safe_save("21_domains_top_registrars.png")

print("Domains EDA complete (6 charts).")

# ============================================================================
# 4. MALICIOUS IPs
# ============================================================================
print("\n" + "=" * 60)
print("4. MALICIOUS IPs EDA")
print("=" * 60)

# 4.1 Threat Severity
fig, ax = plt.subplots(figsize=(7, 5))
ips_sev = ips["Threat_Severity"].value_counts()
order = [s for s in severity_order if s in ips_sev.index]
ax.pie(ips_sev[order].values, labels=order, autopct="%1.1f%%",
       colors=sns.color_palette(PALETTE), startangle=90)
ax.set_title("IP Threat Severity Distribution")
safe_save("22_ips_severity.png")

# 4.2 Top Countries (malicious IP origin)
country_counts = ips["Country"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(country_counts)), country_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(country_counts)))
ax.set_yticklabels(country_counts.index)
ax.invert_yaxis()
ax.set_title("Top 15 Countries Hosting Malicious IPs"); ax.set_xlabel("Count")
count_label(ax)
safe_save("23_ips_top_countries.png")

# 4.3 Continent distribution
cont_counts = ips["Continent"].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(cont_counts.values, labels=cont_counts.index, autopct="%1.1f%%",
       colors=sns.color_palette(PALETTE), startangle=90)
ax.set_title("Malicious IPs by Continent")
safe_save("24_ips_continents.png")

# 4.4 Reputation Score distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(ips["Reputation_Score"].dropna(), bins=30, color=sns.color_palette(PALETTE)[0], edgecolor="white", alpha=0.8)
ax.axvline(ips["Reputation_Score"].median(), color="#E57373", linestyle="--", linewidth=2,
           label=f"Median: {ips['Reputation_Score'].median():.0f}")
ax.set_title("IP Reputation Score Distribution"); ax.set_xlabel("Reputation Score"); ax.set_ylabel("Count")
ax.legend()
safe_save("25_ips_reputation_score.png")

# 4.5 Threat Label distribution
label_counts = ips["Threat_Label"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(label_counts)), label_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(label_counts)))
ax.set_yticklabels(label_counts.index)
ax.invert_yaxis()
ax.set_title("Top 15 IP Threat Labels"); ax.set_xlabel("Count")
count_label(ax)
safe_save("26_ips_threat_labels.png")

# 4.6 Threat Category distribution
cat_counts = ips["Threat_Category"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(cat_counts)), cat_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(cat_counts)))
ax.set_yticklabels(cat_counts.index)
ax.invert_yaxis()
ax.set_title("Top 15 IP Threat Categories"); ax.set_xlabel("Count")
count_label(ax)
safe_save("27_ips_threat_categories.png")

# 4.7 TOR Node analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

tor_counts = ips["TOR_Node"].value_counts()
axes[0].pie(tor_counts.values, labels=["Non-TOR", "TOR Exit Node"], autopct="%1.1f%%",
            colors=sns.color_palette(PALETTE)[:2], startangle=90)
axes[0].set_title("TOR Exit Node Distribution")

tor_sev = pd.crosstab(ips["Threat_Severity"], ips["TOR_Node"])
tor_sev.plot(kind="bar", ax=axes[1], color=sns.color_palette(PALETTE)[:2], edgecolor="white")
axes[1].set_title("TOR Node vs Threat Severity"); axes[1].set_ylabel("Count")
safe_save("28_ips_tor_analysis.png")

# 4.8 Reputation Score vs Malicious Votes
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(ips["Reputation_Score"], ips["Malicious_Votes"],
                c=ips["Threat_Severity"].map({"Low": 0, "Medium": 1, "High": 2, "Critical": 3}),
                cmap=SEQUENTIAL, alpha=0.7, edgecolors="none")
ax.set_title("IP Reputation Score vs Malicious Votes"); ax.set_xlabel("Reputation Score"); ax.set_ylabel("Malicious Votes")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Severity (0=Low, 3=Critical)")
safe_save("29_ips_reputation_vs_votes.png")

# 4.9 Top Owners/ASN
owner_counts = ips["Owner"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(owner_counts)), owner_counts.values, color=sns.color_palette(PALETTE))
ax.set_yticks(range(len(owner_counts)))
ax.set_yticklabels(owner_counts.index)
ax.invert_yaxis()
ax.set_title("Top 10 IP Owners"); ax.set_xlabel("Count")
count_label(ax)
safe_save("30_ips_top_owners.png")

print("IPs EDA complete (9 charts).")

# ============================================================================
# 5. CROSS-DATASET CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("5. CROSS-DATASET SUMMARY")
print("=" * 60)

# 5.1 Summary dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# OTX - monthly trend (small)
otx_m = otx.set_index("Created").resample("ME").size()
axes[0, 0].plot(otx_m.index, otx_m.values, color=sns.color_palette(PALETTE)[0], linewidth=1.5)
axes[0, 0].fill_between(otx_m.index, otx_m.values, alpha=0.2, color=sns.color_palette(PALETTE)[0])
axes[0, 0].set_title("OTX: Monthly Threat Pulses"); axes[0, 0].set_ylabel("Count")

# CVE - year bars
yearly = cve.groupby("CVE_Year")["cveID"].count()
yearly = yearly[yearly.index.notna()]
axes[0, 1].bar(yearly.index.astype(int), yearly.values, color=sns.color_palette(PALETTE)[1])
axes[0, 1].set_title("CVE: Vulnerabilities by Year"); axes[0, 1].set_ylabel("Count")

# Domains - severity pie
dom_sev = domains["Threat_Severity"].value_counts()
axes[1, 0].pie(dom_sev.values, labels=dom_sev.index, autopct="%1.1f%%",
               colors=sns.color_palette(PALETTE), startangle=90)
axes[1, 0].set_title("Domains: Threat Severity")

# IPs - severity pie
ips_sev = ips["Threat_Severity"].value_counts()
axes[1, 1].pie(ips_sev.values, labels=ips_sev.index, autopct="%1.1f%%",
               colors=sns.color_palette(PALETTE), startangle=90)
axes[1, 1].set_title("IPs: Threat Severity")

plt.suptitle("Cybersecurity Datasets - EDA Dashboard", fontsize=16, fontweight="bold", y=1.01)
safe_save("31_summary_dashboard.png")

# 5.2 Correlation heatmap for numeric columns
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Domains numeric correlation
dom_num = domains.select_dtypes(include=[np.number])
if not dom_num.empty and dom_num.shape[1] >= 3:
    sns.heatmap(dom_num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes[0], square=True, linewidths=0.5)
    axes[0].set_title("Domain Numeric Correlations")

# IPs numeric correlation
ips_num = ips.select_dtypes(include=[np.number])
if not ips_num.empty and ips_num.shape[1] >= 3:
    sns.heatmap(ips_num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=axes[1], square=True, linewidths=0.5)
    axes[1].set_title("IP Numeric Correlations")

safe_save("32_correlation_heatmaps.png")

print("Cross-dataset EDA complete (2 charts).")

# ============================================================================
# Final Summary Statistics Print
# ============================================================================
print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(f"\nTotal charts generated: 32")
print(f"Charts saved to: {CHART_DIR}")
print(f"\nDatasets analyzed:")
print(f"  OTX Threat Intel:   {len(otx)} records, {len(otx.columns)} columns")
print(f"  CVE Vulnerabilities: {len(cve)} records, {len(cve.columns)} columns")
print(f"  Malicious Domains:   {len(domains)} records, {len(domains.columns)} columns")
print(f"  Malicious IPs:       {len(ips)} records, {len(ips.columns)} columns")

if len(all_malware) > 0:
    print(f"\nTop 5 Malware Families: {malware_counts.head(5).to_dict()}")
if len(all_attacks) > 0:
    print(f"Top 5 ATT&CK Techniques: {attack_counts.head(5).to_dict()}")
if len(all_ind) > 0:
    print(f"Top 5 Targeted Industries: {ind_counts.head(5).to_dict()}")
