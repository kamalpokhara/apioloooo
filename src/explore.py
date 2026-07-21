import pandas as pd

file_path = "../data/processed/market_data_clean_v1.parquet"

df = pd.read_parquet(file_path)
