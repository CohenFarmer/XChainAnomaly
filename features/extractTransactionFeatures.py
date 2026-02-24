#creates balanced dataset of malicious and non malicious transactions
import pandas as pd

df = pd.read_csv('data/datasets/labeled_cross_chain_transactions_3.csv', low_memory=False)

#samples malicious and non malicious transactions
malicious = df[df['label'] > 0].copy()
non_malicious = df[df['label'] == 0].sample(n=50000, random_state=40)

#converts to binary labels
malicious['label'] = 1

dataset = pd.concat([malicious, non_malicious], ignore_index=True).sample(frac=1.0, random_state=40)

print(f"Malicious: {len(malicious)}, Non-malicious: {len(non_malicious)}, Total: {len(dataset)}")
print(f"Label distribution:\n{dataset['label'].value_counts().sort_index()}")

#drops unnecessary columns
out_df = dataset.drop(columns=['id', 'src_transaction_hash', 'dst_transaction_hash', 'src_to_address', 'dst_to_address', 'dst_from_address', 'src_contract_address', 'dst_contract_address', 'depositor', 'bridge_name'])


print(out_df.head())

out_df.to_csv('features/datasets/cross_chain_labeled_transactions_balanced_50k_v3.csv', index=False)