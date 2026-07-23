import pandas as pd
import numpy as np
from features import load_clean_data, get_src_cols

d = load_clean_data()   # loads v2
src_cols = get_src_cols(d)
imp_cols = [c for c in ['india', 'china', 'bhutan'] if c in src_cols]

d['min_price'] = d['min_price'].replace(0, np.nan)
d['max_price'] = d['max_price'].replace(0, np.nan)

before = len(d)
d = d.dropna(subset=['min_price', 'max_price'])
print(f"Dropped {before - len(d)} additional rows (zero-as-missing min/max)")
# ---- drop rows genuinely missing min/max price (thin, scattered — confirmed safe) ----
def scale01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

# ---- signal 1: price volatility (within-month spread relative to avg) ----
d['volatility'] = (d['max_price'] - d['min_price']) / d['avg_price']

# ---- signal 2: import dependency (already have it, reuse) ----
# d['import_share'] already exists from v2

# ---- signal 3: source concentration ----
d['concentration'] = 1 / d['n_sources'].replace(0, np.nan)

# ---- signal 4: volume instability (CV of total_sources per product, across months) ----
vol_stats = d.groupby('product_name')['total_sources'].agg(['mean', 'std'])
vol_stats['volume_cv'] = np.where(vol_stats['mean'] > 0, vol_stats['std'] / vol_stats['mean'], np.nan)
d = d.merge(vol_stats[['volume_cv']], on='product_name', how='left')

# ---- risk score: average across AVAILABLE signals per row, not forcing all four ----
signal_cols = ['volatility', 'import_share', 'concentration', 'volume_cv']
scaled = pd.DataFrame({c: scale01(d[c]) for c in signal_cols})
d['risk_score'] = scaled.mean(axis=1, skipna=True)   # skipna: rows missing one signal still get a score from the rest

d['risk'] = pd.qcut(d['risk_score'], 3, labels=['Low', 'Medium', 'High'])

print(d[signal_cols + ['risk_score', 'risk']].describe(include='all'))
print(d['risk'].value_counts())

d.to_parquet('../data/processed/market_data_clean_v3.parquet', index=False)
print("Saved v3, shape:", d.shape)