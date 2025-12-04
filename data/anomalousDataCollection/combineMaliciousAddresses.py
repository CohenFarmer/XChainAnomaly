import pandas as pd
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

combined_df = pd.read_csv(basepath + 'combined_malicious_addresses.csv')
tc_interaction = pd.read_csv(basepath + 'cross_chain_transactions_tornado_interacted.csv')
tc_one_hop = pd.read_csv(basepath + 'cross_chain_transactions_tornado_one_hop_interacted.csv')

tc_one_hop = tc_one_hop.rename(columns={'tornado_one_hop_interacted_address': 'tornado_interacted_address'})
tc_interaction = pd.concat([tc_interaction, tc_one_hop], ignore_index=True)
tc_interaction = tc_interaction.drop_duplicates(subset=['tornado_interacted_address'])
tc_interaction['label'] = 4 

final_df = pd.concat([combined_df, tc_interaction.rename(columns={'tornado_interacted_address': 'address'})], ignore_index=True)
final_df = final_df.drop_duplicates(subset=['address'])
final_df.to_csv(basepath + 'final_combined_malicious_addresses.csv', index=False)