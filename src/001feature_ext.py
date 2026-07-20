import pandas as pd, numpy as np

all_months_file = "../data/orginal_data/all_months_clean.csv"
d = pd.read_csv(all_months_file)

d.columns = [c.lower() for c in d.columns]
d = d.sort_values(['product_name', 'month_idx']).reset_index(drop=True)

exclude = ['product_name','category','bs_year','bs_month','month_idx','month_name',
           'volume','min_price','max_price','avg_price','unit','unit_canonical',
           'unit_changed','total_amount','volume_equals','total_sources',
           'reconciliation_gap','import_share','n_months_present','is_balanced']
# all sources
src_cols = [c for c in d.columns if c not in exclude]
# imported sources 
imp_cols = [c for c in ['india', 'china', 'bhutan'] if c in src_cols]

def add_derived_features(df, src_cols=src_cols, imp_cols=imp_cols):
    df = df.copy()
    df['total_sources'] = df[src_cols].sum(axis=1)
    total_safe = df['total_sources'].replace(0, np.nan)

    for c in imp_cols:
        df[c + '_share'] = (df[c] / total_safe).fillna(0)
    df['import_share'] = df[imp_cols].sum(axis=1) / total_safe
    df['domestic_share'] = 1 - df['import_share'].fillna(0)
    df['import_share'] = df['import_share'].fillna(0)
    
    # no. of contributed soruces 
    df['n_sources'] = (df[src_cols] > 0).sum(axis=1)
    # calcualates  Herfindahl index 
    df['herfindahl'] = sum((df[c] / total_safe).fillna(0) ** 2 for c in src_cols)
    # cyclical encoding (circular )
    df['m_sin'] = np.sin(2 * np.pi * df['bs_month'] / 12)
    df['m_cos'] = np.cos(2 * np.pi * df['bs_month'] / 12)
    return df

def add_lags(df):
    df = df.sort_values(['product_name', 'month_idx']).copy()
    df = add_derived_features(df)  # need total_sources first
    g = df.groupby('product_name')
    df['avg_price_lag1'] = g['avg_price'].shift(1)
    df['volume_lag1']    = g['total_sources'].shift(1)   # your call: total_sources not volume
    return df

d = add_lags(d)

def rebuild(df):
    return add_derived_features(df)  # scenario engine calls only this — no lags on shocked data

# print(d[['india_share','china_share','bhutan_share','import_share']].describe())
# print(d[imp_cols + ['total_sources']].isna().sum())
# print(d['reconciliation_gap'].ne(0).sum())   # should be 23 if your data matches the doc

# print(d[['avg_price_lag1','volume_lag1']].isna().sum() )

print("Total Before Fix: ",d['product_name'].nunique())
# print(sorted(d['product_name'].unique()))

name_fixes = {
    'Dragonfruits': 'Dragon_Fruits',
    'Gunduruk': 'Gundruk',
    'Sponge_Groud': 'Sponge_Gourd',
    'Tomato-Big': 'Tomato_Big',
    'Orange (Sweet)': 'Orange_Sweet',
    
    'Sajiwan_Swigan': 'Sajiwan',
    'Brocauli': 'Broccoli',
}
d['product_name'] = d['product_name'].replace(name_fixes)

unique_products = d["product_name"].unique()

print(f"Total After Fix: {len(unique_products)}")
# print(sorted(unique_products))

# pd.set_option("display.max_rows", None)
# product_months = (
#     d.groupby("product_name")["month_idx"]
#       .nunique()
#       .sort_values()
# )

# print(product_months)

# check how many products have zero supply in any month

# zero_supply = d[d['total_sources'] == 0]
# print("Zero Supply: ", len(zero_supply))
# print(zero_supply[['product_name','month_idx']])

# zero_supply = d[d['total_sources'] == 0]
# print(zero_supply[['product_name','month_idx','volume','avg_price']].head(15))



d.to_parquet('../data/processed/market_data_clean_v1.parquet', index=False)

# import os
# import matplotlib.pyplot as plt

# # Count how many months each product appears in
# product_months = (
#     d.groupby("product_name")["month_idx"]
#       .nunique()
# )

# # Count products by month coverage
# coverage_summary = (
#     product_months
#     .value_counts()
#     .sort_index(ascending=False)
# )

# # Create reports folder
# os.makedirs("../reports", exist_ok=True)

# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))

# bars = ax.bar(
#     coverage_summary.index.astype(str),
#     coverage_summary.values
# )

# ax.set_title("Distribution of Products by Number of Months Observed")
# ax.set_xlabel("Number of Months Product Appears")
# ax.set_ylabel("Number of Products")

# # Add values above bars
# for bar, value in zip(bars, coverage_summary.values):
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,
#         value + 0.5,
#         str(value),
#         ha="center",
#         va="bottom"
#     )

# # Add 8+ months threshold
# # The bars are ordered: 10, 9, 8, 7, ..., 1
# threshold_position = list(coverage_summary.index).index(8)

# ax.axvline(
#     x=threshold_position + 0.5,
#     linestyle="--",
#     label="8-month threshold"
# )

# ax.legend()
# ax.grid(axis="y", alpha=0.3)

# plt.tight_layout()

# # Save
# plt.savefig(
#     "../reports/product_month_coverage.png",
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()