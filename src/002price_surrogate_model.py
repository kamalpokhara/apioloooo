import pandas as pd
import lightgbm as lgb
import numpy as np

file_path = "../data/processed/market_data_clean_v1.parquet"
d = pd.read_parquet(file_path)
d = d.drop(columns=['naew'])
d.columns = [c.lower() for c in d.columns]
d = d.sort_values(['product_name', 'month_idx']).reset_index(drop=True)   # moved up, before split
print("File read, Total Rows: ", len(d))

train = d[d['month_idx'] <= 8].copy()
valid = d[d['month_idx'] == 9].copy()
test  = d[d['month_idx'] == 10].copy()
print("\n Train Val Test Split: ", len(train), len(valid), len(test))

exclude = ['product_name','category','bs_year','bs_month','month_idx','month_name',
           'volume','min_price','max_price','avg_price','unit','unit_canonical',
           'unit_changed','total_amount','volume_equals','total_sources',
           'reconciliation_gap','import_share','domestic_share','n_months_present','is_balanced',
           'india_share','china_share','bhutan_share','n_sources','herfindahl',
           'm_sin','m_cos','avg_price_lag1','volume_lag1']
src_cols = [c for c in d.columns if c not in exclude]
imp_cols = [c for c in ['india', 'china', 'bhutan'] if c in src_cols]

feature_cols = (
    ['product_name', 'category', 'unit', 'm_sin', 'm_cos', 'n_sources',
     'herfindahl', 'import_share', 'domestic_share', 'n_months_present',
     'india_share', 'china_share', 'bhutan_share', 'avg_price_lag1']
    + src_cols
)
assert len(feature_cols) == len(set(feature_cols)), "duplicate columns in feature_cols"
print(len(src_cols), "raw source columns — should be 33")

target_col = 'avg_price'


print("\n Feature Columns: ", feature_cols)
print("\n Target Column: ", target_col)

# # Diagonsing 'naew' column
# print("\n Checking for 'naew' column...")

# # Check how many non-zero or non-null entries it has
# print(d['naew'].describe())
# print("Non-zero count:", (d['naew'] > 0).sum())

# Cast categoricals properly LightGBM needs this explicit, don't let it infer

cat_cols = ['product_name', 'category', 'unit']
for df_ in [train, valid, test]:
    for c in cat_cols:
        df_[c] = df_[c].astype('category')

print("\n",train[['product_name','category','unit']].dtypes)

# TRAIN on train set, month 1-8 
X_train, y_train = train[feature_cols].reset_index(drop=True), np.log1p(train[target_col]).reset_index(drop=True).values
X_valid, y_valid = valid[feature_cols].reset_index(drop=True), np.log1p(valid[target_col]).reset_index(drop=True).values
X_test,  y_test  = test[feature_cols].reset_index(drop=True),  np.log1p(test[target_col]).reset_index(drop=True).values

print("X_train columns:", X_train.columns.tolist())
print("Categorical features:", cat_cols)
print("Missing:", set(cat_cols) - set(X_train.columns))

import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def objective(trial):
    params = {
        'objective': 'regression_l1',
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'verbosity': -1,
        'random_state': 42
    }
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, categorical_feature=cat_cols, free_raw_data=False)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    pred = booster.predict(X_valid, num_iteration=booster.best_iteration)
    mae = mean_absolute_error(y_valid, pred)
    print(f"trial {trial.number}: MAE={mae:.4f}  params={params}")
    return mae


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("\nBEST:", study.best_params)
print("BEST valid MAE (log scale):", study.best_value)

# ALL TRIALS AS SORTED TABLE
trials_df = study.trials_dataframe()
print(trials_df[['number','value','params_num_leaves','params_learning_rate',
                  'params_subsample','params_reg_lambda']].sort_values('value'))

# ---- final retrain with best hyperparameters, evaluated properly ----
best_params = study.best_params
best_params.update({'objective': 'regression_l1', 'verbosity': -1, 'random_state': 42})

train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set, categorical_feature=cat_cols, free_raw_data=False)

booster = lgb.train(
    best_params, train_set, num_boost_round=1000,
    valid_sets=[valid_set],
    callbacks=[lgb.early_stopping(50)]
)

pred_valid_price = np.expm1(booster.predict(X_valid, num_iteration=booster.best_iteration))
actual_valid_price = valid[target_col].values

mae_valid_real = mean_absolute_error(actual_valid_price, pred_valid_price)
print("Real valid MAE (price terms):", mae_valid_real)

persist_valid = valid['avg_price_lag1'].fillna(train['avg_price'].mean())
mae_persist_valid = mean_absolute_error(actual_valid_price, persist_valid)
print("Persistence MAE (valid):", mae_persist_valid)

print('\navg_price_lag1' in feature_cols)
print('volume_lag1' in feature_cols)
print("\n Feature Columns: ", feature_cols)

""" OLD CODE: LightGBM model training and evaluation code is commented out below."""
# price_model = lgb.LGBMRegressor(
#     objective='regression_l1',   # MAE-based, more robust to outliers than L2
#     num_leaves=31,
#     learning_rate=0.05,
#     n_estimators=800,
#     subsample=0.9,
#     reg_lambda=1.0,
#     random_state=42
# )

# price_model.fit(
#     X_train, y_train,
#     eval_set=[(X_valid, y_valid)],
#     categorical_feature=cat_cols,
#     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
# )

# # Evaluate on validation set i.e month 9
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# pred_log = price_model.predict(X_valid)
# pred_price = np.expm1(pred_log)
# actual_price = valid[target_col].values

# mae = mean_absolute_error(actual_price, pred_price)

# #  **0.5 basically same as np.sqrt(), or math.sqrt() 

# rmse = mean_squared_error(actual_price, pred_price) ** 0.5
# r2 = r2_score(actual_price, pred_price)
# mape = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100

# print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  MAPE: {mape:.2f}%  R2: {r2:.3f}")

# # Baselines — PERSISTENCE 

# # baseline 1: persistence (this month = last month's price)
# persist_pred = test['avg_price_lag1'].fillna(train['avg_price'].mean())
# mae_persist = mean_absolute_error(actual_price, persist_pred)

# # baseline 2: product mean (from train set)
# prod_mean = train.groupby('product_name')['avg_price'].mean()
# mean_pred = test['product_name'].map(prod_mean).fillna(train['avg_price'].mean())
# mae_mean = mean_absolute_error(actual_price, mean_pred)

# print(f"Persistence MAE: {mae_persist:.2f}   Product-mean MAE: {mae_mean:.2f}   Model MAE: {mae:.2f}")