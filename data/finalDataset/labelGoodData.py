import pandas as pd
import numpy as np

def normalize_address(address: str) -> str:
        address = address.strip().lower()
        if address.startswith("0x"):
            address = address[2:]
        return address

def search_by_sorted_address(target: str, malicious_data: np.ndarray) -> bool:
    key = normalize_address(target)
    idx = np.searchsorted(malicious_data, key)
    return idx < len(malicious_data) and malicious_data[idx] == key

transactions = pd.read_parquet('data/datasets/cross_chain_unified.parquet', engine='pyarrow')
transactions = transactions.sample(frac=1, random_state=42)
transactions = transactions.reset_index(drop=True)

malicious_addresses = pd.read_csv('data/datasets/final_combined_malicious_addresses.csv')
malicious_data = malicious_addresses['address'].astype(str).map(normalize_address).to_numpy()

final_df= pd.DataFrame(columns=['address', 'blockchain', 'label'], dtype=object)

for index, row in transactions.head(250000).iterrows():
    depositor = normalize_address(row['depositor'])
    recipient = normalize_address(row['recipient'])

    if (search_by_sorted_address(depositor, malicious_data) or 
        search_by_sorted_address(recipient, malicious_data)):
        continue

    final_df = pd.concat([final_df, pd.DataFrame([[row['depositor'].lower(), row['src_blockchain'], 0]], columns=['address', 'blockchain', 'label'])], ignore_index=True)
    final_df = pd.concat([final_df, pd.DataFrame([[row['recipient'].lower(), row['dst_blockchain'], 0]], columns=['address', 'blockchain', 'label'])], ignore_index=True)

print(len(final_df))
final_df.drop_duplicates(subset=['address'], inplace=True)
print(len(final_df))
final_df.to_csv('data/datasets/good_addresses.csv', index=False)

    
