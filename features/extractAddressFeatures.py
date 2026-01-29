import pandas as pd
from data.dataExtraction.alchemyGetAddressTransfers import getAddressTransfers
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
from data.helpers.alchemyAPI import alchemyClient
from datetime import datetime, timezone
from threading import Lock

feature_cache = {}
cache_lock = Lock()

"""good_df = pd.read_csv('data/datasets/good_addresses.csv')
malicious_df = pd.read_csv('data/datasets/malicious_address_tornado_5000.csv')
malicious_df['blockchain'] = "ethereum"

cols = ['address', 'blockchain', 'label']

supported_chains = ['ethereum', 'arbitrum', 'optimism', 'polygon']

malicious_df = malicious_df.sample(frac=1, random_state=42).reset_index(drop=True)

malicious_df = malicious_df[cols]
malicious_df = malicious_df.head(2000)

good_df = good_df.sample(frac=1, random_state=42).reset_index(drop=True)
good_df = good_df.head(30000)"""
df = pd.read_csv('features/datasets/cctx_transfer_features.csv')

df['source_index'] = pd.to_numeric(df['source_index'], errors='coerce').astype('Int64')

df_src = df[df['role'] == 'src_from']
df_recipient = df[df['role'] == 'recipient']

# Updated to use new balanced dataset with 2600 malicious + 50k non-malicious
final_df = pd.read_csv('features/datasets/cross_chain_labeled_transactions_balanced_50k_v3.csv')
max_index = len(final_df)

df_src_missing_index = set(range(0, max_index)) - set(df_src['source_index'].drop_duplicates().tolist())
print(f"Missing src_from indices: {len(df_src_missing_index)}")
df_recipient_missing_index = set(range(0, max_index)) - set(df_recipient['source_index'].drop_duplicates().tolist())
print(f"Missing recipient indices: {len(df_recipient_missing_index)}")
supported_chains = ['ethereum', 'arbitrum', 'optimism', 'polygon']

colls = ['source_index', 'role', 'address', 'transfers_from_count', 'transfers_to_count', 'total_transfers', 'total_value', 'avg_value', 'earliest_timestamp', 'latest_timestamp', 'avg_time_between_transactions', 'min_tx_val', 'max_tx_val', 'count_tx_over_1_eth', 'count_tx_over_10_eth', 'unique_from_addresses', 'unique_to_addresses', 'time_active', 'avg_tx_freq', 'label']

def collect_transfer_data(transfer, total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp):
    raw_ts = transfer['metadata']['blockTimestamp'] if 'metadata' in transfer and 'blockTimestamp' in transfer['metadata'] else None
    if raw_ts:
        if (earliest_timestamp is None) or (raw_ts < earliest_timestamp):
            earliest_timestamp = raw_ts
        if (latest_timestamp is None) or (raw_ts > latest_timestamp):
            latest_timestamp = raw_ts

    value = transfer['value']
    if value:
        total_value += value
        if value > 1:
            count_tx_over_1_eth += 1
        if value > 10:
            count_tx_over_10_eth += 1
        if value < min_tx_val:
            min_tx_val = value
        if value > max_tx_val:
            max_tx_val = value
    return total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp

def compute_address_features(address: str, label: int, chain: str, source_index: int, role: str):
    key = (str(address).lower(), str(chain).lower())
    with cache_lock:
        cached = feature_cache.get(key)
    if cached:
        data = dict(cached)
        data["source_index"] = int(source_index)
        data["role"] = role
        data["label"] = label
        return data
    client = getAddressTransfers()
    from_transfers = client.fetch_transfers(chain, "from", from_address=address)
    to_transfers = client.fetch_transfers(chain, "to", to_address=address)

    transfers_from_count = from_transfers['count']
    transfers_to_count = to_transfers['count']
    unique_from_addresses = set()
    unique_to_addresses = set()

    total_transfers = transfers_from_count + transfers_to_count

    total_value = 0
    count_tx_over_1_eth = 0
    count_tx_over_10_eth = 0
    min_tx_val = float('inf')
    max_tx_val = 0
    earliest_timestamp = None
    latest_timestamp = None

    for transfer in from_transfers['transfers']:
        total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp = collect_transfer_data(
            transfer, total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp
        )
        transfer_to_address = transfer['to']
        unique_to_addresses.add(transfer_to_address)
    
    for transfer in to_transfers['transfers']:
        total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp = collect_transfer_data(
            transfer, total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp
        )
        transfer_from_address = transfer['from']
        unique_from_addresses.add(transfer_from_address)
    
    if earliest_timestamp is None:
        earliest_timestamp = 0
    if latest_timestamp is None:
        latest_timestamp = 0
    else:
        earliest_timestamp = int(datetime.fromisoformat(earliest_timestamp.replace("Z","+00:00")).timestamp() * 1000)
        latest_timestamp = int(datetime.fromisoformat(latest_timestamp.replace("Z","+00:00")).timestamp() * 1000)
    if (min_tx_val == float('inf')):
        min_tx_val = 0
    avg_value = total_value / total_transfers if total_transfers > 0 else 0
    time_active  = latest_timestamp - earliest_timestamp
    avg_time_between_transactions = (time_active) / total_transfers if total_transfers > 0 else 0
    result = {
        "source_index": int(source_index),
        "role": role,
        "address": address,
        "transfers_from_count": transfers_from_count,
        "transfers_to_count": transfers_to_count,
        "total_transfers": total_transfers,
        "total_value": total_value,
        "avg_value": avg_value,
        "earliest_timestamp": earliest_timestamp,
        "latest_timestamp": latest_timestamp,
        "avg_time_between_transactions": avg_time_between_transactions,
        "min_tx_val": float(min_tx_val),
        "max_tx_val": float(max_tx_val),
        "count_tx_over_1_eth": count_tx_over_1_eth,
        "count_tx_over_10_eth": count_tx_over_10_eth,
        "unique_from_addresses": len(unique_from_addresses),
        "unique_to_addresses": len(unique_to_addresses),
        "time_active": time_active,
        "avg_tx_freq": (total_transfers/(time_active/86400000)) if time_active > 0 else 0,
        "label": label,
    }

    base = dict(result)
    base["source_index"] = 0
    base["role"] = ""
    
    base["label"] = 0
    with cache_lock:
        feature_cache[key] = base

    return result
    

if __name__ == "__main__":
    work_list = []
    for idx, r in final_df.iterrows():
        if idx in df_src_missing_index:
            work_list.append((r['src_from_address'].lower(), r['label'], idx, r['src_blockchain'], 'src_from'))
    for idx, r in final_df.iterrows():
        if idx in df_recipient_missing_index:
            work_list.append((r['recipient'].lower(), r['label'], idx, r['dst_blockchain'], 'recipient'))
    max_workers = 16
    per_future_wait_timeout = 10
    max_retries = 3
    base_backoff_seconds = 2

    results = []
    buffer = []
    checkpoint_every = 20
    output_path = 'features/datasets/cctx_transfer_features.csv'
    first_write = True
    retry_counts = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {}
        for address, label, idx, chain, role in work_list:
            if chain.lower() not in supported_chains:
                print(f'Skipping unsupported chain {chain} for address {address}')
                continue
            f = executor.submit(compute_address_features, address, label, chain, idx, role)
            pending[f] = (address, label, idx, chain, role)
            retry_counts[(address, idx)] = 0

        while pending:
            done, not_done = wait(list(pending.keys()), timeout=per_future_wait_timeout, return_when=FIRST_COMPLETED)

            for f in done:
                address, label, idx, chain, role = pending.pop(f)
                try:
                    data = f.result()
                    results.append(data)
                    buffer.append(data)
                    print(f'Completed features for address: {address}, role: {role}, label: {label}, index: {idx}, chain: {chain}')
                    print(len(buffer))
                    
                    if len(buffer) >= checkpoint_every:
                        df_chunk = pd.DataFrame(buffer, columns=colls)
                        df_chunk.to_csv(output_path, mode='a', header=first_write, index=False)
                        first_write = False
                        buffer.clear()
                except Exception as exc:
                    key = (address, idx)
                    if retry_counts[key] < max_retries:
                        retry_counts[key] += 1
                        backoff = base_backoff_seconds * (2 ** (retry_counts[key] - 1))
                        print(f'Address {address} failed ({exc}). Retry {retry_counts[key]}/{max_retries} after {backoff}s')
                        time.sleep(backoff)
                        nf = executor.submit(compute_address_features, address, label, chain, idx, role)
                        pending[nf] = (address, label, idx, chain, role)
                    else:
                        print(f'Skipping {address} after {max_retries} failures')

            if not done and not_done:
                f_timeout = next(iter(not_done))
                address, label, idx, chain, role = pending.pop(f_timeout)
                key = (address, idx)
                if retry_counts[key] < max_retries:
                    retry_counts[key] += 1
                    backoff = base_backoff_seconds * (2 ** (retry_counts[key] - 1))
                    print(f'Timeout waiting for {address}. Retry {retry_counts[key]}/{max_retries} after {backoff}s')
                    time.sleep(backoff)
                    nf = executor.submit(compute_address_features, address, label, chain, idx, role)
                    pending[nf] = (address, label, idx, chain, role)
                else:
                    print(f'Skipping {address} after {max_retries} timeouts')

    if buffer:
        df_chunk = pd.DataFrame(buffer, columns=colls)
        df_chunk.to_csv(output_path, mode='a', header=first_write, index=False)
        first_write = False
        buffer.clear()

    final_df = pd.DataFrame(results, columns=colls)

    final_df.to_csv(output_path, index=False)
    print(f'Saved features to {output_path}')