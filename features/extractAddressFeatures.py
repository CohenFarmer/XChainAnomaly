import pandas as pd
from data.dataExtraction.alchemyGetAddressTransfers import getAddressTransfers
from datetime import datetime, timezone

good_df = pd.read_csv('data/datasets/good_addresses.csv')
malicious_df = pd.read_csv('data/datasets/final_combined_malicious_addresses.csv')
malicious_df['blockchain'] = "ethereum"

cols = ['address', 'blockchain', 'label']

malicious_df = malicious_df.sample(frac=1, random_state=42).reset_index(drop=True)

malicious_df = malicious_df[cols]
print(malicious_df['label'].value_counts())
print(len(malicious_df))
malicious_df = malicious_df.head(500)

good_df = good_df[cols]

eth_only = pd.DataFrame(columns=cols)

for index, row in good_df.iterrows():
    if row['blockchain'].lower() == 'ethereum':
        eth_only = pd.concat([eth_only, pd.DataFrame([[row['address'], row['blockchain'], row['label']]], columns=cols)], ignore_index=True)
print(len(eth_only))

dataframes = [malicious_df, eth_only]

colls = ['address', 'transfers_from_count', 'transfers_to_count', 'total_transfers', 'total_value', 'avg_value', 'earliest_timestamp', 'latest_timestamp', 'label']
"""final_df = pd.DataFrame(columns=colls)
for df in dataframes:
    for index, row in df.iterrows():
        all_transfers_from = getAddressTransfers().fetch_transfers("ethereum", "from", from_address=row['address'])
        all_transfers_to = getAddressTransfers().fetch_transfers("ethereum", "to", to_address=row['address'])

        transfers_from_count = all_transfers_from['count']
        transfers_to_count = all_transfers_to['count']

        total_transfers = transfers_from_count + transfers_to_count

        total_value = 0

        earliest_timestamp = None
        latest_timestamp = None
        for transfer in all_transfers_from['transfers']:
            if (earliest_timestamp is None) or (transfer['metadata']['blockTimestamp'] < earliest_timestamp):
                earliest_timestamp = transfer['metadata']['blockTimestamp']
            if (latest_timestamp is None) or (transfer['metadata']['blockTimestamp'] > latest_timestamp):
                latest_timestamp = transfer['metadata']['blockTimestamp']
            value = transfer['value']
            if value:
                total_value += int(value)
    
        for transfer in all_transfers_to['transfers']:
            if (earliest_timestamp is None) or (transfer['metadata']['blockTimestamp'] < earliest_timestamp):
                earliest_timestamp = transfer['metadata']['blockTimestamp']
            if (latest_timestamp is None) or (transfer['metadata']['blockTimestamp'] > latest_timestamp):
                latest_timestamp = transfer['metadata']['blockTimestamp']
            value = transfer['value']
            if value:
                total_value += int(value)
        if earliest_timestamp is None:
            earliest_timestamp = 0
        if latest_timestamp is None:
            latest_timestamp = 0
        else:
            earliest_timestamp = int(datetime.fromisoformat(earliest_timestamp.replace("Z","+00:00")).timestamp() * 1000)
            latest_timestamp = int(datetime.fromisoformat(latest_timestamp.replace("Z","+00:00")).timestamp() * 1000)
        avg_value = total_value / total_transfers if total_transfers > 0 else 0
        print("Address: ", index, row['address'], transfers_from_count, transfers_to_count, total_value, avg_value, earliest_timestamp, latest_timestamp)
        newRow = pd.DataFrame([[row['address'], transfers_from_count, transfers_to_count, total_transfers, total_value, avg_value, earliest_timestamp, latest_timestamp, row['label']]], columns=colls)
        final_df = pd.concat([final_df, newRow], ignore_index=True)

final_df.to_csv('features/datasets/address_transfer_features_eth.csv', index=False)
"""