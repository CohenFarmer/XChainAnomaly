import pandas as pd

df = pd.read_csv('data/datasets/labeled_cross_chain_transactions.csv')

neg = df[df['label'] == 0]
pos = df[df['label'] == 1]
neg_sample = neg.sample(n=min(50000, len(neg)), random_state=40)
dataset = pd.concat([pos, neg_sample], ignore_index=True).sample(frac=1.0, random_state=40)

out_df = dataset.drop(columns=['id', 'src_transaction_hash', 'dst_transaction_hash', 'src_to_address', 'dst_to_address', 'dst_from_address', 'src_contract_address', 'dst_contract_address', 'depositor', 'bridge_name'])


print(out_df.head())

out_df.to_csv('features/datasets/cross_chain_labeled_transactions_balanced_50k.csv', index=False)