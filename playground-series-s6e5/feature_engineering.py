import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


# ===== 1. Load Data =====
print('Loading data...')
train = pd.read_csv('E:/py_project/playground-series-s6e5/train.csv')
test = pd.read_csv('E:/py_project/playground-series-s6e5/test.csv')
sub = pd.read_csv('E:/py_project/playground-series-s6e5/sample_submission.csv')

# 2023 anomaly: keep original label for potential later use, but we need to fix it
# Observation: 2023 has pit rate ~0.96% vs other years ~27-29%. 
# Strategy: Create is_2023 flag AND consider 2023 data may have labeling issues
train['_is_2023'] = (train['Year'] == 2023).astype(int)
test['_is_2023'] = (test['Year'] == 2023).astype(int)

# Combine for feature engineering
train_len = len(train)
target = train['PitNextLap'].copy()
del train['PitNextLap']

all_data = pd.concat([train, test], axis=0, ignore_index=True)

print(f'Train: {train_len}, Test: {len(test)}, Combined: {len(all_data)}')


# ===== 2. Feature Engineering =====

# --- 2a. TyreLife normalised by compound ---
compound_tyre_avg = all_data.groupby('Compound')['TyreLife'].agg(['mean', 'std', 'max']).to_dict('index')
all_data['TyreLife_Compound_Ratio'] = all_data.apply(
    lambda row: row['TyreLife'] / max(compound_tyre_avg.get(row['Compound'], {}).get('max', 1), 1), axis=1
)
all_data['TyreLife_Zscore'] = all_data.apply(
    lambda row: (row['TyreLife'] - compound_tyre_avg.get(row['Compound'], {}).get('mean', 1)) /
                 max(compound_tyre_avg.get(row['Compound'], {}).get('std', 1), 1), axis=1
)
# TyreLife percentile within compound
for comp in all_data['Compound'].unique():
    mask = all_data['Compound'] == comp
    all_data.loc[mask, 'TyreLife_Compound_Rank'] = all_data.loc[mask, 'TyreLife'].rank(pct=True)

# --- 2b. Degradation rate ---
all_data['Degradation_Rate'] = all_data['Cumulative_Degradation'] / (all_data['TyreLife'] + 1)
all_data['Deg_Abs'] = np.abs(all_data['Cumulative_Degradation'])

# --- 2c. Is this driver's last stint in this race? (approximation) ---
race_stint_max = all_data.groupby(['Race', 'Year', 'Driver'])['Stint'].transform('max')
all_data['Is_Last_Stint'] = (all_data['Stint'] == race_stint_max).astype(int)

# --- 2d. LapNumber features ---
all_data['LapNumber_Squared'] = all_data['LapNumber'] ** 2
all_data['LapNumber_Log'] = np.log1p(all_data['LapNumber'])

# --- 2e. Race progress features ---
all_data['RaceProgress_Squared'] = all_data['RaceProgress'] ** 2
all_data['RaceProgress_Bucket'] = pd.cut(all_data['RaceProgress'], bins=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                                          labels=[0, 1, 2, 3, 4, 5]).astype(int)

# --- 2f. Position_Change features ---
all_data['Position_Change_Abs'] = np.abs(all_data['Position_Change'])
all_data['Position_Change_Squared'] = all_data['Position_Change'] ** 2
all_data['Is_Position_Stable'] = (all_data['Position_Change'] == 0).astype(int)

# --- 2g. Position features ---
all_data['Position_Bucket'] = pd.cut(all_data['Position'], bins=[0, 3, 5, 10, 15, 20],
                                      labels=[0, 1, 2, 3, 4]).astype(int)

# --- 2h. LapTime features ---
lt_col = 'LapTime (s)'
all_data['LapTime_Log'] = np.log1p(all_data[lt_col])
# Flag for extremely slow laps (could be safety car or already in pit lane)
all_data['Is_Very_Slow_Lap'] = (all_data[lt_col] > 130).astype(int)
all_data['Is_Extreme_Lap'] = (all_data[lt_col] > 200).astype(int)
# LapTime × TyreLife interaction
all_data['LapTime_TyreLife'] = all_data[lt_col] * all_data['TyreLife'] / 100

# --- 2i. Interaction features ---
# Stint × TyreLife
all_data['Stint_TyreLife'] = all_data['Stint'] * all_data['TyreLife']
# Stint × Compound
all_data['Stint_Hard'] = all_data['Stint'] * (all_data['Compound'] == 'HARD').astype(int)
# Position × RaceProgress
all_data['Pos_RaceProgress'] = all_data['Position'] * all_data['RaceProgress']
# Is fresh tyre (first 3 laps of stint)
all_data['Is_Fresh_Tyre'] = (all_data['TyreLife'] <= 3).astype(int)
# Is old tyre (above compound median + 50%)
for comp in all_data['Compound'].unique():
    mask = all_data['Compound'] == comp
    median_life = compound_tyre_avg.get(comp, {}).get('mean', 10) * 0.75
    all_data.loc[mask, 'Is_Old_Tyre'] = (all_data.loc[mask, 'TyreLife'] > median_life).astype(int)

# --- 2j. Driver features ---
all_data['Driver_Is_Named'] = all_data['Driver'].str.match(r'^[A-Z]{3}$').astype(int)
all_data['Driver_Char_Count'] = all_data['Driver'].str.len()

# --- 2k. Race-level aggregated features ---
# Avg TyreLife per race-stage
for feature in ['TyreLife', 'LapTime (s)', 'Cumulative_Degradation']:
    col_key = feature.replace(' (s)', '').replace(' ', '_')
    all_data[f'Race_Avg_{col_key}'] = all_data.groupby(['Race', 'Year'])[feature].transform('mean')
    all_data[f'Race_Std_{col_key}'] = all_data.groupby(['Race', 'Year'])[feature].transform('std')

# --- 2l. Laps remaining approximation ---
race_max_laps = all_data.groupby(['Race', 'Year'])['LapNumber'].transform('max')
all_data['Laps_Remaining'] = race_max_laps - all_data['LapNumber']
all_data['Laps_Remaining_Ratio'] = all_data['Laps_Remaining'] / race_max_laps

# --- 2m. Cumulative_Degradation direction ---
all_data['Deg_Is_Negative'] = (all_data['Cumulative_Degradation'] < 0).astype(int)

# --- 2n. Year interactions with key features ---
all_data['Year_TyreLife'] = all_data['Year'] * all_data['TyreLife']
all_data['Year_Stint'] = all_data['Year'] * all_data['Stint']
all_data['Year_RaceProgress'] = all_data['Year'] * all_data['RaceProgress']


# ===== 3. Encode Categorical Features =====

# Compound: ordinal encoding by observed hardness
compound_order = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
all_data['Compound_Encoded'] = all_data['Compound'].map(compound_order)

# Driver: Label encode (high cardinality, 887 unique)
le_driver = LabelEncoder()
all_data['Driver_Encoded'] = le_driver.fit_transform(all_data['Driver'])

# Race: Label encode (26 unique)
le_race = LabelEncoder()
all_data['Race_Encoded'] = le_race.fit_transform(all_data['Race'])

# Year: keep as int, but also create one-hot-ish features for specific interaction
# Year encoded as ordinal
all_data['Year_Encoded'] = all_data['Year'] - 2022


# ===== 4. Drop original string columns and id =====
drop_cols = ['Driver', 'Compound', 'Race']
all_data.drop(columns=drop_cols, inplace=True)


# ===== 5. Split back =====
train_processed = all_data.iloc[:train_len].copy()
test_processed = all_data.iloc[train_len:].copy()
train_processed['PitNextLap'] = target.values
test_processed['id'] = sub['id'].values  # ensure correct id order

# Drop id from train (not needed for training)
train_processed.drop(columns=['id'], inplace=True)
test_ids = test_processed['id'].copy()
test_processed.drop(columns=['id'], inplace=True)

print(f'\nTrain shape: {train_processed.shape}')
print(f'Test shape: {test_processed.shape}')
print(f'Train features: {train_processed.drop(columns=["PitNextLap"]).columns.tolist()}')

# Verify no NaN
train_nan = train_processed.isnull().sum().sum()
test_nan = test_processed.isnull().sum().sum()
print(f'\nTrain NaN count: {train_nan}')
print(f'Test NaN count: {test_nan}')

# Save
train_processed.to_csv('E:/py_project/playground-series-s6e5/train_processed.csv', index=False)
test_processed.to_csv('E:/py_project/playground-series-s6e5/test_processed.csv', index=False)

print('\nFeature engineering complete. Files saved.')

# Print feature importance preview via correlation
print('\n=== TOP 15 FEATURES BY CORRELATION WITH TARGET ===')
corr_with_target = train_processed.corr()['PitNextLap'].drop('PitNextLap').abs().sort_values(ascending=False)
print(corr_with_target.head(20).to_string())
