import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

df = pd.read_parquet('data/datasets/cross_chain_unified.parquet', engine='pyarrow')

tornado_df = pd.read_csv('data/datasets/tornado_cash_interacted_addresses_eth.csv')

print(df.head())
print(df.columns)
all_tornado_interactions = set()
all_one_hop_tornado_interactions = set()

for index, row in df.iterrows():
    if row['src_blockchain'].lower() == 'ethereum':
        if row['depositor'].lower() in tornado_df['tornado_interacted_address'].values:
            print("Found src address interaction with Tornado Cash:", row['depositor'], index)
            all_tornado_interactions.add(row['depositor'].lower())
            all_one_hop_tornado_interactions.add(row['recipient'].lower())
    if row['dst_blockchain'].lower() == 'ethereum':
        if row['recipient'].lower() in tornado_df['tornado_interacted_address'].values:
            print("Found dst address interaction with Tornado Cash:", row['recipient'], index)
            all_tornado_interactions.add(row['recipient'].lower())
            all_one_hop_tornado_interactions.add(row['depositor'].lower())

tcOnehop = pd.DataFrame(list(all_one_hop_tornado_interactions))
tcOnehop.columns = ['tornado_one_hop_interacted_address']
tcOnehop.to_csv('data/datasets/cross_chain_transactions_tornado_one_hop_interacted.csv', index=False)

tc = pd.DataFrame(list(all_tornado_interactions))
tc.columns = ['tornado_interacted_address']
tc.to_csv('data/datasets/cross_chain_transactions_tornado_interacted.csv', index=False)