#labels cross chain transactions based on malicious address involvement
import pandas as pd
import data.anomalousDataCollection.tornadoCashInteraction as tornadoCashInteraction

#label meanings: 0=clean 1=phishing 2=exploit 3=sanctioned 4=tornado

df = pd.read_parquet('data/datasets/cross_chain_unified.parquet', engine='pyarrow')
blockchains = set(['ethereum', 'polygon', 'arbitrum', 'optimism'])

df_with_only_supported_chains = df[
    df['src_blockchain'].isin(blockchains) & df['dst_blockchain'].isin(blockchains)
].reset_index(drop=True)
print("Filtered to supported chains:", len(df_with_only_supported_chains))

src_col = 'src_from_address' if 'src_from_address' in df_with_only_supported_chains.columns else 'depositor'
recip_col = 'recipient' if 'recipient' in df_with_only_supported_chains.columns else 'recipient'
src_lower = df_with_only_supported_chains[src_col].fillna('').str.lower()
recip_lower = df_with_only_supported_chains[recip_col].fillna('').str.lower()

#loads malicious addresses with labels
malicious_df = pd.read_csv('data/datasets/malicious_address_all.csv')
address_to_label = dict(zip(
    malicious_df['address'].astype(str).str.lower(),
    malicious_df['label']
))

#adds tornado addresses not in main list
tornado_df = pd.read_csv('data/datasets/tornado_cash_interacted_addresses_eth.csv')
for addr in tornado_df['tornado_interacted_address'].astype(str).str.lower():
    if addr not in address_to_label:
        address_to_label[addr] = 4

#returns label for transaction based on src and recipient
def get_label(src, recip):
    if src in address_to_label:
        return address_to_label[src]
    if recip in address_to_label:
        return address_to_label[recip]
    return 0

labels = [get_label(s, r) for s, r in zip(src_lower, recip_lower)]
df_with_only_supported_chains['label'] = labels

#prints label distribution
label_counts = df_with_only_supported_chains['label'].value_counts().sort_index()
print("\nLabel distribution:")
print("0 (Non-malicious):", label_counts.get(0, 0))
print("1 (Phishing):", label_counts.get(1, 0))
print("2 (Exploit/Heist):", label_counts.get(2, 0))
print("3 (Sanctioned):", label_counts.get(3, 0))
print("4 (Tornado Cash):", label_counts.get(4, 0))

df_with_only_supported_chains.to_csv('data/datasets/labeled_cross_chain_transactions.csv', index=False)
print("\nSaved to data/datasets/labeled_cross_chain_transactions_3.csv")