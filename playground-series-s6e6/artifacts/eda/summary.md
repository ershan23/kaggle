# EDA Summary - Stellar Classification

## Dataset Size
- Train: 577,347 rows x 12 cols
- Test: 247,435 rows x 11 cols
- Missing values: None

## Target Distribution
- GALAXY: 377,480 (65.4%)
- QSO: 117,143 (20.3%)
- STAR: 82,724 (14.3%)

## Key Findings
### 1. Redshift is the strongest single discriminator
  - GALAXY: median=0.4820, std=0.3091
  - QSO: median=1.7989, std=1.0697
  - STAR: median=0.0565, std=0.0645
  - Negative redshift: 8957 rows (1.551%)

### 2. Photometric bands (u,g,r,i,z) are highly correlated
  - Pairwise correlation range: [0.444, 0.969]
  - Color indices (differences) provide independent information

### 3. Categorical features have strong class association
  - spectral_type:
    - A/F: dominant class = QSO (50.4%)
    - G/K: dominant class = GALAXY (56.8%)
    - M: dominant class = GALAXY (95.0%)
    - O/B: dominant class = QSO (71.1%)
  - galaxy_population:
    - Blue_Cloud: dominant class = QSO (42.0%)
    - Red_Sequence: dominant class = GALAXY (90.3%)

### 4. Train/Test distribution shift (KS test)
  - alpha: KS=0.0021, p=4.35e-01
  - delta: KS=0.0014, p=8.98e-01
  - u: KS=0.0009, p=9.99e-01
  - g: KS=0.0018, p=6.55e-01
  - r: KS=0.0017, p=7.16e-01
  - i: KS=0.0024, p=2.61e-01
  - z: KS=0.0028, p=1.43e-01
  - redshift: KS=0.0023, p=3.34e-01
  - No significant distribution shift detected

### 5. Sky distribution (alpha/delta)
  - Objects are spread across the observable sky (SDSS footprint)
  - No obvious spatial clustering by class

## Feature Engineering Recommendations
1. Color indices (u-g, g-r, r-i, i-z) as primary features
2. Redshift log-transform + negative flag
3. Magnitude aggregations (mean, std, range)
4. Cyclic encoding for alpha (RA)
5. Categorical interaction: spectral_type x galaxy_population
