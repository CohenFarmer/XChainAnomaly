import pandas as pd
from data.helpers import alchemyAPI
basepath = 'data/datasets/'

"""sanctioned_data = pd.read_csv(basepath + 'sanctioned_addresses_ETH.txt', sep=" ", header=None)
etherscan_malicious_labels = pd.read_csv(basepath + 'etherscan_malicious_labels.csv')
heist_label = pd.read_csv(basepath + 'Heist label-etherscan.csv')
phishing_label = pd.read_csv(basepath + 'phishing_label.csv')

column_names = ['address', 'label']

complete_df = pd.DataFrame(columns=column_names)

for index, row in etherscan_malicious_labels.iterrows():
    label = 0
    if row['data_source'] == 'etherscan-phish-hack-list':
        label = 1
    else:
        label = 2
    address = row['banned_address']
    complete_df = pd.concat([complete_df, pd.DataFrame([[address, label]], columns=column_names)], ignore_index=True)

for index, row in sanctioned_data.iterrows():
    address = row[0]
    label = 3
    complete_df = pd.concat([complete_df, pd.DataFrame([[address, label]], columns=column_names)], ignore_index=True)

for index, row in heist_label.iterrows():
    address = row['Address']
    label = 2
    complete_df = pd.concat([complete_df, pd.DataFrame([[address, label]], columns=column_names)], ignore_index=True)

for index, row in phishing_label.iterrows():
    address = row['address']
    label = 1
    if row['tag'][:13] == 'Fake_Phishing':
        complete_df = pd.concat([complete_df, pd.DataFrame([[address, label]], columns=column_names)], ignore_index=True)

complete_df = complete_df['address'].str.lower().to_frame().join(complete_df['label'])

#no overlapping addresses with different labels, 
#for example 0xabcd -> exploit and 0xabcd -> sanctioned, so it's
#fine to drop duplicates without worrying about labels
complete_df = complete_df.drop_duplicates(subset=['address'])

complete_df.to_csv(basepath + 'combined_malicious_address.csv', index=False)"""

alchemyAPI_client = alchemyAPI.alchemyClient("eth-mainnet")

combined_df = pd.read_csv(basepath + 'combined_malicious_addresses.csv')
tc_addresses = pd.read_csv(basepath + 'tornado_cash_interacted_addresses_eth.csv')
#tc_one_hop = pd.read_csv(basepath + 'cross_chain_transactions_tornado_one_hop_interacted.csv')

#tc_one_hop = tc_one_hop.rename(columns={'tornado_one_hop_interacted_address': 'tornado_interacted_address'})
#tc_interaction = pd.concat([tc_interaction, tc_one_hop], ignore_index=True)
tc_addresses = tc_addresses.drop_duplicates(subset=['tornado_interacted_address'])
tc_addresses['label'] = 4 

final_df = pd.concat([combined_df, tc_addresses.rename(columns={'tornado_interacted_address': 'address'})], ignore_index=True)
final_df = final_df.drop_duplicates(subset=['address'])

from concurrent.futures import ThreadPoolExecutor, as_completed

final_final_df = pd.DataFrame(columns=final_df.columns)

def check_and_collect(row):
    address = row['address']
    try:
        if not alchemyAPI_client.is_contract(address):
            return address, row['label']
    except Exception:
        return None
    return None

max_workers = 8
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(check_and_collect, row): idx for idx, row in final_df.iterrows()}
    for future in as_completed(futures):
        print(len(final_final_df))
        result = future.result()
        if result is not None:
            address, label = result
            final_final_df = pd.concat([final_final_df, pd.DataFrame([[address, label]], columns=final_df.columns)], ignore_index=True)

final_final_df = final_final_df.sort_values(by='address').reset_index(drop=True)
final_final_df.to_csv(basepath + 'malicious_address_all.csv', index=False)