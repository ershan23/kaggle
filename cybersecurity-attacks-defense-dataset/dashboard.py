# -*- coding: utf-8 -*-
"""
Cybersecurity Datasets Dashboard
Streamlit-based interactive dashboard with insights & recommendations.
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import matplotlib.ticker as mticker
from datetime import datetime

st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide", page_icon=":shield:")

BASE_DIR = r"E:\py_project\Cybersecurity Attacks & Defense Dataset 2026"
CLEAN_DIR = os.path.join(BASE_DIR, "cleaned")
CHART_DIR = os.path.join(BASE_DIR, "eda_charts")
MODEL_DIR = os.path.join(BASE_DIR, "model_outputs")

# Style
plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "axes.titlesize": 12, "axes.labelsize": 10})
PALETTE = sns.color_palette("Set2").as_hex()


# ============================================================================
# CACHE DATA LOADING
# ============================================================================
@st.cache_data
def load_all_data():
    otx = pd.read_csv(os.path.join(CLEAN_DIR, "1_otx_threat_intel_clean.csv"))
    cve = pd.read_csv(os.path.join(CLEAN_DIR, "2_cve_vulnerabilities_clean.csv"))
    domains = pd.read_csv(os.path.join(CLEAN_DIR, "3_malicious_domains_clean.csv"))
    ips = pd.read_csv(os.path.join(CLEAN_DIR, "4_malicious_ips_clean.csv"))

    # Fix dates
    otx["Created"] = pd.to_datetime(otx["Created"], errors="coerce")
    cve["dateAdded"] = pd.to_datetime(cve["dateAdded"], errors="coerce")
    cve["dueDate"] = pd.to_datetime(cve["dueDate"], errors="coerce")

    # Parse list columns
    def _parse(val):
        if pd.isna(val) or str(val).strip() in ["", "[]"]:
            return []
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return [x.strip() for x in str(val).split(",") if x.strip()]

    for col in ["Tags_Clean", "Industries_Clean", "Malware_Families_Clean", "Attack_IDs_Clean"]:
        if col in otx.columns:
            otx[col] = otx[col].apply(_parse)
    cve["cwes_list"] = cve["cwes_list"].apply(_parse)
    cve["CVE_Year"] = pd.to_numeric(cve["CVE_Year"], errors="coerce").astype("Int64")
    cve["knownRansomwareCampaignUse"] = cve["knownRansomwareCampaignUse"].fillna(False).astype(bool)
    domains["Threat_Severity"] = domains["Threat_Severity"].fillna("Unknown")
    ips["Threat_Severity"] = ips["Threat_Severity"].fillna("Unknown")
    ips["TOR_Node"] = ips["TOR_Node"].fillna(False).astype(bool)
    return otx, cve, domains, ips

otx, cve, domains, ips = load_all_data()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title(":shield: Navigation")
page = st.sidebar.radio("Go to", ["Overview", "OTX Threat Intel", "CVE Vulnerabilities",
                                   "Malicious Domains", "Malicious IPs", "Model Results", "Recommendations"])

# ============================================================================
# HELPER
# ============================================================================
def plot_to_st(fig):
    st.pyplot(fig)
    plt.close()

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if page == "Overview":
    st.title("Cybersecurity Attacks & Defense Dataset 2026")
    st.markdown("### Dataset Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("OTX Threat Pulses", f"{len(otx):,}", delta="2,340 unique")
    with col2:
        st.metric("CVE Vulnerabilities", f"{len(cve):,}", delta=f"{cve['CVE_Year'].max()} max year")
    with col3:
        st.metric("Malicious Domains", len(domains), delta=f"{domains['Threat_Severity'].value_counts().get('High', 0)} High Risk")
    with col4:
        st.metric("Malicious IPs", len(ips), delta=f"{ips['Threat_Severity'].value_counts().get('High', 0)} High Risk")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Threat Landscape Summary")
        fig, ax = plt.subplots(figsize=(8, 6))
        datasets = ["OTX\nThreats", "CVE\nVulns", "Malicious\nDomains", "Malicious\nIPs"]
        sizes = [len(otx), len(cve), len(domains), len(ips)]
        colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
        wedges, texts, autotexts = ax.pie(sizes, labels=datasets, autopct="%1.1f%%",
                                           colors=colors, startangle=90,
                                           textprops={"fontsize": 11})
        for t in autotexts:
            t.set_fontweight("bold")
        ax.set_title("Dataset Distribution", fontsize=14, fontweight="bold")
        plot_to_st(fig)

    with col2:
        st.subheader("Threat Severity Comparison")
        sev_order = ["Low", "Medium", "High", "Critical"]
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(3)
        width = 0.35
        dom_sev = [domains["Threat_Severity"].value_counts().get(s, 0) for s in sev_order[:3]]
        ips_sev = [ips["Threat_Severity"].value_counts().get(s, 0) for s in sev_order[:3]]
        ax.bar(x - width/2, dom_sev, width, label="Domains", color=PALETTE[2], edgecolor="white")
        ax.bar(x + width/2, ips_sev, width, label="IPs", color=PALETTE[3], edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(sev_order[:3])
        ax.set_ylabel("Count"); ax.set_title("Threat Severity: Domains vs IPs", fontsize=14, fontweight="bold")
        ax.legend()
        for bar in ax.containers:
            ax.bar_label(bar, fontsize=10, padding=2)
        plot_to_st(fig)

    st.markdown("---")
    st.subheader("Monthly Trend: Threat Intelligence & CVEs")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        otx_m = otx.set_index("Created").resample("ME").size()
        ax.fill_between(otx_m.index, otx_m.values, alpha=0.3, color=PALETTE[0])
        ax.plot(otx_m.index, otx_m.values, marker="o", markersize=3, color=PALETTE[0])
        ax.set_title("OTX Threat Pulses (Monthly)", fontweight="bold")
        ax.set_ylabel("Count")
        plot_to_st(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        cve_m = cve.set_index("dateAdded").resample("ME").size()
        ax.fill_between(cve_m.index, cve_m.values, alpha=0.3, color=PALETTE[1])
        ax.plot(cve_m.index, cve_m.values, marker="o", markersize=3, color=PALETTE[1])
        ax.set_title("CVE Additions (Monthly)", fontweight="bold")
        ax.set_ylabel("Count")
        plot_to_st(fig)

# ============================================================================
# OTX PAGE
# ============================================================================
elif page == "OTX Threat Intel":
    st.title("OTX Threat Intelligence (2,340 records)")
    st.markdown("AlienVault Open Threat Exchange - global threat intelligence pulses")

    tab1, tab2, tab3 = st.tabs(["Malware & Techniques", "Targets", "Temporal"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            all_malware = []
            for f in otx["Malware_Families_Clean"]: all_malware.extend(f)
            mc = pd.Series(all_malware).value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(mc)), mc.values, color=PALETTE[0])
            ax.set_yticks(range(len(mc))); ax.set_yticklabels(mc.index); ax.invert_yaxis()
            ax.set_title("Top 10 Malware Families", fontweight="bold")
            for i, v in enumerate(mc.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

        with col2:
            all_atk = []
            for a in otx["Attack_IDs_Clean"]: all_atk.extend(a)
            ac = pd.Series(all_atk).value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(ac)), ac.values, color=PALETTE[1])
            ax.set_yticks(range(len(ac))); ax.set_yticklabels(ac.index); ax.invert_yaxis()
            ax.set_title("Top 10 ATT&CK Techniques", fontweight="bold")
            for i, v in enumerate(ac.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

        st.markdown("---")
        st.subheader("TLP Distribution")
        fig, ax = plt.subplots(figsize=(7, 3))
        tlp = otx["TLP"].value_counts()
        tlp_colors = {"white": "#A5D6A7", "green": "#66BB6A", "amber": "#FFB74D", "red": "#E57373"}
        ax.bar(tlp.index, tlp.values, color=[tlp_colors.get(k, "#BDBDBD") for k in tlp.index], edgecolor="white")
        ax.set_title("Traffic Light Protocol (TLP)", fontweight="bold"); ax.set_ylabel("Count")
        for bar in ax.containers: ax.bar_label(bar, fontsize=10, padding=2)
        plot_to_st(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            all_ind = []
            for i in otx["Industries_Clean"]: all_ind.extend(i)
            ic = pd.Series(all_ind).value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(ic)), ic.values, color=PALETTE[2])
            ax.set_yticks(range(len(ic))); ax.set_yticklabels(ic.index); ax.invert_yaxis()
            ax.set_title("Top 10 Targeted Industries", fontweight="bold")
            for i, v in enumerate(ic.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

        with col2:
            all_ctry = []
            for c in otx["Countries_Clean"]: all_ctry.extend(c)
            cc = pd.Series(all_ctry).value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(cc)), cc.values, color=PALETTE[3])
            ax.set_yticks(range(len(cc))); ax.set_yticklabels(cc.index); ax.invert_yaxis()
            ax.set_title("Top 10 Targeted Countries", fontweight="bold")
            for i, v in enumerate(cc.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 5))
        otx_daily = otx.set_index("Created").resample("D").size()
        ax.fill_between(otx_daily.index, otx_daily.values, alpha=0.4, color=PALETTE[0])
        ax.plot(otx_daily.index, otx_daily.values, linewidth=1, color=PALETTE[0])
        ax.set_title("OTX Threat Pulses - Daily Trend", fontweight="bold")
        ax.set_ylabel("Count")
        plot_to_st(fig)

        burst_dates = otx_daily.nlargest(5)
        if len(burst_dates) > 0:
            st.subheader("Top 5 Burst Days (Most Pulses)")
            for d, v in burst_dates.items():
                st.write(f"- **{d.strftime('%Y-%m-%d')}**: {v} pulses")

# ============================================================================
# CVE PAGE
# ============================================================================
elif page == "CVE Vulnerabilities":
    st.title("CVE Vulnerabilities (1,585 records)")
    st.markdown("CISA KEV Catalog + NVD - actively exploited vulnerabilities")

    tab1, tab2, tab3 = st.tabs(["Year & Severity", "Vendors & CWEs", "Response Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            yearly = cve.groupby("CVE_Year").agg(
                total=("cveID", "count"),
                ransom=("knownRansomwareCampaignUse", "sum")
            ).dropna()
            fig, ax = plt.subplots(figsize=(8, 5))
            x = yearly.index.astype(int)
            ax.bar(x, yearly["total"], alpha=0.8, label="All CVEs", color=PALETTE[0])
            ax.bar(x, yearly["ransom"], alpha=0.9, label="Ransomware", color="#E57373")
            ax.set_title("CVE by Year (Ransomware Overlay)", fontweight="bold")
            ax.set_xlabel("Year"); ax.set_ylabel("Count"); ax.legend()
            plot_to_st(fig)

        with col2:
            total_ransom = cve["knownRansomwareCampaignUse"].sum()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([len(cve) - total_ransom, total_ransom],
                   labels=["Not Ransomware", "Ransomware-Known"],
                   autopct="%1.1f%%", colors=[PALETTE[0], "#E57373"], startangle=90, explode=(0, 0.05))
            ax.set_title(f"Ransomware Usage ({total_ransom}/{len(cve)} CVEs)", fontweight="bold")
            plot_to_st(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            vc = cve["vendorProject"].value_counts().head(12)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(vc)), vc.values, color=PALETTE[1])
            ax.set_yticks(range(len(vc))); ax.set_yticklabels(vc.index); ax.invert_yaxis()
            ax.set_title("Top 12 Vendors", fontweight="bold")
            for i, v in enumerate(vc.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

        with col2:
            all_cwes = []
            for cw in cve["cwes_list"]: all_cwes.extend(cw)
            cwe_counts = pd.Series(all_cwes).value_counts().head(12)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(cwe_counts)), cwe_counts.values, color=PALETTE[2])
            ax.set_yticks(range(len(cwe_counts))); ax.set_yticklabels(cwe_counts.index); ax.invert_yaxis()
            ax.set_title("Top 12 CWE Weaknesses", fontweight="bold")
            for i, v in enumerate(cwe_counts.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Response days
            cve_r = cve.copy()
            cve_r["response_days"] = (cve_r["dueDate"] - cve_r["dateAdded"]).dt.days
            cve_r = cve_r[cve_r["response_days"].notna() & (cve_r["response_days"] >= 0) & (cve_r["response_days"] < 60)]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(cve_r["response_days"], bins=25, color=PALETTE[0], edgecolor="white", alpha=0.8)
            ax.axvline(cve_r["response_days"].median(), color="#E57373", linestyle="--", linewidth=2,
                       label=f"Median: {cve_r['response_days'].median():.0f} d")
            ax.axvline(14, color="#FFB74D", linestyle="--", linewidth=1.5, label="BOD 22-01: 14 d")
            ax.set_title("Response Window (dateAdded -> dueDate)", fontweight="bold")
            ax.set_xlabel("Days"); ax.set_ylabel("Count"); ax.legend()
            plot_to_st(fig)

        with col2:
            st.subheader("Response Time Stats")
            if len(cve_r) > 0:
                stats = cve_r["response_days"].describe()
                st.metric("Median", f"{stats['50%']:.0f} days")
                st.metric("Mean", f"{stats['mean']:.1f} days")
                st.metric("Min", f"{stats['min']:.0f} days")
                st.metric("Max", f"{stats['max']:.0f} days")
                overdue = (cve_r["response_days"] > 14).sum()
                st.metric("Overdue (>14d)", f"{overdue} ({overdue/len(cve_r)*100:.1f}%)")
            st.caption("BOD 22-01 mandates patching within 14 days")

# ============================================================================
# DOMAINS PAGE
# ============================================================================
elif page == "Malicious Domains":
    st.title("Malicious Domains (162 records)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Domains", len(domains))
    with col2:
        high_count = domains["Threat_Severity"].value_counts().get("High", 0)
        st.metric("High Severity", high_count, delta_color="inverse")
    with col3:
        st.metric("Unique TLDs", domains["TLD"].nunique())

    tab1, tab2 = st.tabs(["Severity & TLDs", "Domain Features"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            sev = domains["Threat_Severity"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.pie(sev.values, labels=sev.index, autopct="%1.1f%%", colors=PALETTE[:4], startangle=90)
            ax.set_title("Threat Severity Distribution", fontweight="bold")
            plot_to_st(fig)

        with col2:
            tld_c = domains["TLD"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(tld_c.index, tld_c.values, color=PALETTE[2], edgecolor="white")
            ax.set_title("Top 10 TLDs", fontweight="bold"); ax.set_ylabel("Count")
            for bar in ax.containers: ax.bar_label(bar, fontsize=9, padding=2)
            plot_to_st(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 5))
            nums_pct = domains["Has_Numbers"].value_counts(normalize=True) * 100
            ax.pie(nums_pct.values, labels=["No Numbers", "Has Numbers"], autopct="%1.1f%%", colors=PALETTE[:2], startangle=90)
            ax.set_title("Has Numbers in Domain", fontweight="bold")
            plot_to_st(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            hyp_pct = domains["Has_Hyphen"].value_counts(normalize=True) * 100
            ax.pie(hyp_pct.values, labels=["No Hyphen", "Has Hyphen"], autopct="%1.1f%%", colors=PALETTE[2:4], startangle=90)
            ax.set_title("Has Hyphen in Domain", fontweight="bold")
            plot_to_st(fig)

        st.subheader("High Severity Domains")
        high_domains = domains[domains["Threat_Severity"] == "High"]
        if len(high_domains) > 0:
            st.dataframe(high_domains[["Domain", "TLD", "Registrar", "Malicious_Votes", "Suspicious_Votes"]],
                         use_container_width=True)
        else:
            st.info("No high-severity domains found")

# ============================================================================
# IPS PAGE
# ============================================================================
elif page == "Malicious IPs":
    st.title("Malicious IPs (200 records)")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("IPs", len(ips))
    with col2:
        high_count = ips["Threat_Severity"].value_counts().get("High", 0)
        st.metric("High Severity", high_count, delta_color="inverse")
    with col3:
        tor_count = ips["TOR_Node"].sum()
        st.metric("TOR Nodes", tor_count)
    with col4:
        st.metric("Countries", ips["Country"].nunique())

    tab1, tab2 = st.tabs(["Geography & Severity", "Threat Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            cont = ips["Continent"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.pie(cont.values, labels=cont.index, autopct="%1.1f%%", colors=PALETTE, startangle=90)
            ax.set_title("IPs by Continent", fontweight="bold")
            plot_to_st(fig)

        with col2:
            country_c = ips["Country"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(country_c)), country_c.values, color=PALETTE[3])
            ax.set_yticks(range(len(country_c))); ax.set_yticklabels(country_c.index); ax.invert_yaxis()
            ax.set_title("Top 10 Countries", fontweight="bold")
            for i, v in enumerate(country_c.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            tc = ips["Threat_Category"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(range(len(tc)), tc.values, color=PALETTE[0])
            ax.set_yticks(range(len(tc))); ax.set_yticklabels(tc.index); ax.invert_yaxis()
            ax.set_title("Top 10 Threat Categories", fontweight="bold")
            for i, v in enumerate(tc.values): ax.text(v+1, i, str(v), va="center", fontsize=9)
            plot_to_st(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(ips["Reputation_Score"].dropna(), bins=20, color=PALETTE[1], edgecolor="white", alpha=0.8)
            ax.axvline(ips["Reputation_Score"].median(), color="#E57373", linestyle="--", linewidth=2,
                       label=f"Median: {ips['Reputation_Score'].median():.0f}")
            ax.set_title("Reputation Score Distribution", fontweight="bold")
            ax.set_xlabel("Reputation Score"); ax.legend()
            plot_to_st(fig)

        st.subheader("High Severity IPs")
        high_ips = ips[ips["Threat_Severity"] == "High"]
        if len(high_ips) > 0:
            st.dataframe(high_ips[["IP", "Country", "Owner", "Threat_Label", "Reputation_Score", "TOR_Node"]],
                         use_container_width=True)

# ============================================================================
# MODEL RESULTS PAGE
# ============================================================================
elif page == "Model Results":
    st.title("Model Performance Summary")

    st.markdown("### 4 Classification Tasks with Multiple Models")
    st.markdown("Models: Logistic Regression | Random Forest | XGBoost | LightGBM")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CVE Ransomware Prediction")
        st.metric("Best Model", "Random Forest", delta="F1 Macro: 0.67")
        st.markdown("""
        - **1,585 samples, binary** (20% positive)
        - **Recall (Ransomware): 52%** — catches half of ransomware-used CVEs
        - Top features: `remote code execution`, `elevation of privilege`, CVE year
        - ROC-AUC: **0.75** (all models)
        """)

        st.subheader("Domain Threat Severity")
        st.metric("Best Model", "XGBoost", delta="F1 Macro: 1.00")
        st.markdown("""
        - **162 samples, 3-class** (89% Low)
        - Warning: small sample - likely overfitting
        - Recommendation: collect more data for production use
        """)

    with col2:
        st.subheader("OTX Industry Classification")
        st.metric("Best Model", "XGBoost", delta="F1 Weighted: 0.56")
        st.markdown("""
        - **2,256 samples, 15 classes** (54% Unknown)
        - **Challenge**: "Unknown" dominates, minority classes ~0 F1
        - Improvement: label propagation, entity extraction from descriptions
        """)

        st.subheader("IP Threat Severity")
        st.metric("Best Model", "XGBoost", delta="F1 Macro: 1.00")
        st.markdown("""
        - **200 samples, 3-class** (87% Low)
        - Warning: small sample - likely overfitting
        - Top features: Reputation_Score, Malicious_Votes, TOR_Node
        """)

    # Load and display model comparison data
    st.markdown("---")
    st.subheader("Detailed Metrics")

    model_files = {
        "CVE Ransomware": "task1_cve_model_comparison.csv",
        "OTX Industry": "task2_otx_model_comparison.csv",
        "Domain Severity": "task3_domains_model_comparison.csv",
        "IP Severity": "task4_ips_model_comparison.csv",
    }

    selected_task = st.selectbox("Select Task", list(model_files.keys()))
    csv_path = os.path.join(MODEL_DIR, model_files[selected_task])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"Model CSV not found: {csv_path}")

# ============================================================================
# RECOMMENDATIONS PAGE
# ============================================================================
else:
    st.title("Actionable Recommendations")

    st.markdown("""
    Based on the exploratory data analysis and modeling of **4,312 cybersecurity records** across
    4 datasets, here are prioritized recommendations for threat intelligence, vulnerability
    management, and SOC operations.
    """)

    # Priority 1: Critical
    st.markdown("---")
    st.header(":red_circle: Priority 1 — Immediate Actions")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patch the Top 5 CVEs Under Active Ransomware Exploitation")
        cve_ransom = cve[cve["knownRansomwareCampaignUse"] == True]
        st.markdown(f"""
        - **{len(cve_ransom)} CVEs** are known to be used in ransomware campaigns
        - **Top affected vendors**: Microsoft, Cisco, Apple, Apache, Google
        - **Most targeted CWEs**:
          - **CWE-416** (Use After Free) —  common in browser/OS exploits
          - **CWE-787** (Out-of-bounds Write) —  memory corruption
          - **CWE-22** (Path Traversal) —  file access attacks
        - **Action**: Prioritize patching CVEs with `knownRansomwareCampaignUse=True` within 14 days per BOD 22-01
        """)

    with col2:
        st.subheader("Harden Against Top ATT&CK Techniques")
        top_techniques = {"T1027": "Obfuscated Files or Information",
                          "T1082": "System Information Discovery",
                          "T1140": "Deobfuscate/Decode Files or Information",
                          "T1105": "Ingress Tool Transfer",
                          "T1059": "Command and Scripting Interpreter"}
        st.markdown("**Top 5 Most Frequent ATT&CK Techniques:**")
        for tid, name in top_techniques.items():
            st.markdown(f"- **{tid}**: {name}")
        st.markdown("""
        - **T1027** (996 occurrences, 42.6% of pulses) — deploy AMSI bypass protection, PowerShell logging
        - **T1082** (749 occurrences) — monitor system enumeration via endpoint telemetry
        - **T1059** — restrict PowerShell execution policy, enable ScriptBlock logging
        """)

    # Priority 2: Strategic
    st.markdown("---")
    st.header(":orange_circle: Priority 2 — Strategic Improvements")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Industry-Specific Defense Playbooks")
        st.markdown("""
        Based on OTX targeting data:
        - **Government** (547 pulses, 23.4%) — APT-focused defense, supply chain security
        - **Finance** (395 pulses, 16.9%) — anti-phishing, credential theft prevention, banking trojan detection
        - **Technology** (343 pulses, 14.7%) — software supply chain, CI/CD pipeline security
        - **Defense** (186 pulses) — air-gapped network monitoring, insider threat detection
        - **Manufacturing** (150 pulses) — OT/IoT security, ransomware resilience

        **Action**: Develop tailored threat models for your industry vertical.
        """)

    with col2:
        st.subheader("Malicious Infrastructure Monitoring")
        st.markdown(f"""
        **Domains (162 records):**
        - **89.5% Low severity** —  but **7 High-severity domains** need immediate blocking
        - **Top TLDs**: `.com`, `.org`, `.top`, `.xyz` —  monitor new registrations in these TLDs
        - **Warning signs**: domains with numbers + hyphens + shady registrars = higher risk

        **IPs (200 records):**
        - **87% Low**, **12 High-severity** IPs —  add to firewall deny lists
        - **Top threat labels**: `clean` (unrated), `malware`, `phishing`, `proxy`
        - **TOR nodes (3%)**: 6 TOR exit nodes detected —  monitor for anonymized C2 traffic
        - **Geographic concentration**: EU (48.4%), AS (32.3%) —  consider geo-blocking for high-risk regions

        **Action**: Automate daily ingestion of Domain/IP feeds into SIEM/firewall blocklists.
        """)

    # Priority 3: Long-term
    st.markdown("---")
    st.header(":large_blue_circle: Priority 3 — Long-term Investments")

    st.markdown("""
    #### Model Operationalization
    - **CVE Ransomware Classifier** (Random Forest, F1=0.67): Deploy as a triage tool to flag newly disclosed CVEs for ransomware risk within seconds — integrate with your SIEM/TIP pipeline
    - **OTX Industry Classifier**: Currently limited by 54% "Unknown" labels — invest in **entity extraction from threat descriptions** (NER using spaCy/transformers) to auto-populate industries — could push F1 from 0.27 → 0.60+
    - **Domain/IP Severity**: Collect more labeled data (target: 1,000+ per dataset) before production deployment

    #### Data Quality Improvements
    - **Enrich OTX**: Cross-reference Pulse_IDs with MITRE ATT&CK groups, CVE references, and VirusTotal Intelligence
    - **Enrich CVE**: Add CVSS v3.1 vector strings, EPSS scores, and public exploit availability from ExploitDB/GitHub
    - **Enrich Domains/IPs**: Add passive DNS data, SSL certificate chains, and URL path analysis for better classification

    #### Emerging Threats to Watch
    - **Cobalt Strike** (62 pulses) remains the #1 post-exploitation tool — monitor for beacon configurations
    - **Lumma Stealer** (46 pulses) and **AsyncRAT** (43 pulses) — rising infostealer/RAT threats
    - **Post-quantum ransomware** (Kyber) —  first real-world hybrid encryption malware already observed
    - **ClickFix phishing** —  novel social engineering technique targeting end users with fake CAPTCHA/OTP flows
    """)

    # KPI Summary Table
    st.markdown("---")
    st.subheader("Summary KPI Matrix")

    kpi_data = {
        "KPI": [
            "Total Records Analyzed",
            "CVEs with Known Ransomware Use",
            "Top Malware Family",
            "Top ATT&CK Technique",
            "Top Targeted Industry",
            "Median CVE Patch Window",
            "High/Critical Domains",
            "High/Critical IPs",
            "TOR Exit Nodes Detected",
            "Model: CVE Ransomware F1",
        ],
        "Value": [
            "4,312",
            f"317 ({317/1585*100:.1f}%)",
            "Cobalt Strike (62 pulses)",
            "T1027 - Obfuscation (996 pulses)",
            "Government (547 pulses)",
            f"{cve.assign(d=(cve['dueDate']-cve['dateAdded']).dt.days)['d'].median():.0f} days",
            f"{domains['Threat_Severity'].value_counts().get('High', 0)}",
            f"{ips['Threat_Severity'].value_counts().get('High', 0)}",
            str(int(ips['TOR_Node'].sum())),
            "0.67 (Random Forest)",
        ],
        "Action": [
            "Continuous monitoring",
            "Patch within 14 days",
            "Deploy Cobalt Strike detection rules",
            "Enable AMSI + PowerShell logging",
            "Tailor government-sector defenses",
            "Reduce via automated prioritization",
            "Add to DNS sinkhole",
            "Add to firewall deny list",
            "Monitor TOR exit traffic",
            "Deploy as CVE triage tool",
        ],
    }
    kpi_df = pd.DataFrame(kpi_data)
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Dashboard powered by Streamlit | Data: AlienVault OTX, CISA KEV, NVD, VirusTotal | Analysis Date: " + datetime.now().strftime("%Y-%m-%d"))


if __name__ == "__main__":
    pass
