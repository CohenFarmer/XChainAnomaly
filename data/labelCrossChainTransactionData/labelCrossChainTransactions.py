import pandas as pd
import data.anomalousDataCollection.tornadoCashInteraction as tornadoCashInteraction

df = pd.read_parquet('data/datasets/cross_chain_unified.parquet', engine='pyarrow')
blockchains = set(['ethereum', 'polygon', 'arbitrum', 'optimism'])

df_with_only_supported_chains = df[
    df['src_blockchain'].isin(blockchains) & df['dst_blockchain'].isin(blockchains)
].reset_index(drop=True)
print("Filtered to supported chains:", len(df_with_only_supported_chains))

tornadoCashInteraction_instance = tornadoCashInteraction.tornadoInteraction()
complete_labeled_addresses = pd.DataFrame(columns=df_with_only_supported_chains.columns)
complete_labeled_addresses['label'] = pd.Series(dtype=int)

src_col = 'src_from_address' if 'src_from_address' in df_with_only_supported_chains.columns else 'depositor'
recip_col = 'recipient' if 'recipient' in df_with_only_supported_chains.columns else 'recipient'
src_lower = df_with_only_supported_chains[src_col].fillna('').str.lower()
recip_lower = df_with_only_supported_chains[recip_col].fillna('').str.lower()

# Match against EOA addresses that interacted with Tornado Cash
tornado_interacted_df = pd.read_csv('data/datasets/tornado_cash_interacted_addresses_eth.csv')
interacted = set(tornado_interacted_df['tornado_interacted_address'].astype(str).str.lower())

#tornado_interacted_df = pd.read_csv('data/datasets/malicious_address_all.csv')
#interacted = set(tornado_interacted_df['address'].astype(str).str.lower())

is_malicious = src_lower.isin(interacted) | recip_lower.isin(interacted)

df_malicious_cctx = df_with_only_supported_chains[is_malicious]
df_non_malicious_cctx = df_with_only_supported_chains[~is_malicious]

print("Found malicious cross-chain transactions:", len(df_malicious_cctx))
print("Found non-malicious cross-chain transactions:", len(df_non_malicious_cctx))

complete_labeled_addresses = pd.concat([
    df_malicious_cctx.assign(label=1),
    df_non_malicious_cctx.assign(label=0)
], ignore_index=True)

complete_labeled_addresses.to_csv('data/datasets/labeled_cross_chain_transactions.csv', index=False)

    

