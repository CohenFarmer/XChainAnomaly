#combines malicious addresses from multiple sources and filters out contracts
import pandas as pd
from data.helpers import alchemyAPI
basepath = 'data/datasets/'

alchemyAPI_client = alchemyAPI.alchemyClient("eth-mainnet")

combined_df = pd.read_csv(basepath + 'combined_malicious_addresses.csv')
tc_addresses = pd.read_csv(basepath + 'tornado_cash_interacted_addresses_eth.csv')

tc_addresses = tc_addresses.drop_duplicates(subset=['tornado_interacted_address'])
tc_addresses['label'] = 4 

final_df = pd.concat([combined_df, tc_addresses.rename(columns={'tornado_interacted_address': 'address'})], ignore_index=True)
final_df = final_df.drop_duplicates(subset=['address'])

from concurrent.futures import ThreadPoolExecutor, as_completed

final_final_df = pd.DataFrame(columns=final_df.columns)

#checks if address is not a contract and returns it
def check_and_collect(row):
    address = row['address']
    try:
        if not alchemyAPI_client.is_contract(address):
            return address, row['label']
    except Exception:
        return None
    return None

max_workers = 8
#filters out contract addresses in parallel
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