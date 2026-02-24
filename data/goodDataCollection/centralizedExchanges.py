#finds addresses that have interacted with centralized exchanges
import json
import pandas as pd
from data.dataExtraction import alchemyGetAddressTransfers

cexExchangesInteractions = pd.DataFrame(columns=['address', 'blockchain', 'label'])

with open('data/datasets/cexHotWallets.json', 'r') as f:
    data = json.load(f)

#normalizes address to lowercase
def norm(addr: str) -> str:
    return str(addr).strip().lower()

#builds set of cex addresses for fast lookup
cex_set = set()
addr_to_exchange = {}
for ex in data:
    name = ex['name'] if 'name' in ex else (ex['exchange'] if 'exchange' in ex else 'unknown')
    for a in (ex['addresses'] if 'addresses' in ex else []):
        na = norm(a)
        cex_set.add(na)
        addr_to_exchange[na] = name

alchemyGetAddressTransfers_instance = alchemyGetAddressTransfers.getAddressTransfers()

blockchains = ["ethereum"]

good_df = pd.read_csv('data/datasets/good_addresses.csv')
good_df = good_df[good_df['blockchain'].isin(blockchains)].reset_index(drop=True)

for index, row in good_df.iterrows():
    address = row['address']
    chain = row['blockchain']
from concurrent.futures import ThreadPoolExecutor, as_completed

#checks if address has sent to or received from cex
def check_address(row):
    address = row['address']
    chain = row['blockchain']
    label = row['label'] if 'label' in row else None
    try:
        from_resp = alchemyGetAddressTransfers_instance.fetch_transfers(chain, "from", from_address=address)
        to_resp = alchemyGetAddressTransfers_instance.fetch_transfers(chain, "to", to_address=address)
        from_transfers = from_resp["transfers"] if "transfers" in from_resp else []
        to_transfers = to_resp["transfers"] if "transfers" in to_resp else []
        fromAddresses = {norm(t["to"]) for t in from_transfers if ('to' in t and t['to'])}
        toAddresses   = {norm(t["from"]) for t in to_transfers if ('from' in t and t['from'])}
        has_cex_relation = bool(fromAddresses & cex_set) or bool(toAddresses & cex_set)
        if has_cex_relation:
            return address, chain, label
    except Exception:
        return None
    return None

max_workers = 8
#processes addresses in parallel
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(check_address, row): i for i, row in good_df.iterrows()}
    for idx, f in enumerate(as_completed(futures)):
        print(f"Processed {idx + 1} addresses")
        res = f.result()
        if res is not None:
            addr, chain, label = res
            print(f"Address {address} on {chain} has CEX interaction.")
            cexExchangesInteractions = pd.concat(
                [cexExchangesInteractions, pd.DataFrame([[addr, chain, label]], columns=['address','blockchain','label'])],
                ignore_index=True
            )
        if (idx + 1) % 1000 == 0:
            print('Processed', idx + 1, 'good addresses')

cexExchangesInteractions.drop_duplicates(subset=['address','blockchain','label'], inplace=True)
cexExchangesInteractions.to_csv('data/datasets/cex_interacted_addresses_eth.csv', index=False)