import pandas as pd
from data.anomalousDataCollection.tornadoCashInteraction import tornadoInteraction
from data.dataExtraction.alchemyGetAddressTransfers import getAddressTransfers

getTcAddr = tornadoInteraction()
getAddrTx = getAddressTransfers()

df = pd.read_parquet('data/datasets/cross_chain_unified.parquet', engine='pyarrow')

all_tornado_interactions = set()
for index, row in df.iterrows():
    addr = None
    if row['src_blockchain'].lower() == 'ethereum':
        addr = row['depositor'].lower()
    elif row['dst_blockchain'].lower() == 'ethereum':
        addr = row['recipient'].lower()
    else:
        continue
    address_from_transactions = getAddrTx.fetch_transfers('ethereum', 'from', from_address=addr)
    for transfer in address_from_transactions['transfers']:
        to_addr = transfer['to']
        if to_addr is None:
            continue
        if getTcAddr.in_tornado_address(to_addr):
            print("Found One hop src address interaction with Tornado Cash via transactions:", addr, index)
            print("To Address:", to_addr)
            all_tornado_interactions.add(addr)

    address_to_transactions = getAddrTx.fetch_transfers('ethereum', 'to', to_address=addr)
    if (len(address_to_transactions['transfers']) > 1000):
        continue
    for transfer in address_to_transactions['transfers']:
        from_addr = transfer['from']
        if from_addr is None:
            continue
        if getTcAddr.in_tornado_address(from_addr):
            print("Found One hop src address interaction with Tornado Cash via transactions:", addr, index)
            print("From Address:", from_addr)
            all_tornado_interactions.add(addr)

print(len(all_tornado_interactions))