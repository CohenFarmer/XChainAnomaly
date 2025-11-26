import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

df = pd.read_parquet('data/datasets/cross_chain_unified.parquet', engine='pyarrow')

aliases = {"ethereum"}
mask = df["src_blockchain"].str.lower().isin(aliases) | df["dst_blockchain"].str.lower().isin(aliases)

first_idx = mask.idxmax() if mask.any() else None  # idx of first True
first_row = df.loc[first_idx] if first_idx is not None else None
print(first_row['dst_transaction_hash'] if first_row is not None else "No matching row found")