"""
Add new malicious transactions to the enriched dataset.
1. Get all malicious transactions from labeled_cross_chain_transactions_3.csv
2. Fetch on-chain features for their addresses (multi-threaded, saves progress)
3. Predict 5-class probabilities using address model
4. Append to existing enriched dataset
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.dataExtraction.alchemyGetAddressTransfers import getAddressTransfers

PROGRESS_FILE = 'features/datasets/new_address_features_progress.csv'
BATCH_SIZE = 10

# Load existing enriched data
enriched = pd.read_csv('features/datasets/cross_chain_labeled_transactions_enriched_probs.csv')
print(f"Existing enriched: {len(enriched)} rows")
print(f"  Labels: {enriched['label'].value_counts().sort_index().to_dict()}")

# Load new labeled data - get ALL malicious
labeled = pd.read_csv('data/datasets/labeled_cross_chain_transactions_3.csv', low_memory=False)
all_malicious = labeled[labeled['label'] > 0].copy()

print(f"\nTotal malicious in labeled file: {len(all_malicious)} rows")

# Deduplicate: remove transactions already in enriched based on (src_from, recipient, timestamp)
enriched_keys = set(zip(
    enriched['src_from_address'].str.lower(), 
    enriched['recipient'].str.lower(), 
    enriched['src_timestamp']
))
new_malicious = all_malicious[~all_malicious.apply(
    lambda r: (str(r['src_from_address']).lower(), str(r['recipient']).lower(), r['src_timestamp']) in enriched_keys, 
    axis=1
)].copy()

print(f"After deduplication: {len(new_malicious)} new rows to add")
print(f"  By bridge: {new_malicious['bridge_name'].value_counts().to_dict()}")

# Convert label to binary (1 = malicious)
new_malicious['label'] = 1

# Unique addresses to fetch
src_addresses = new_malicious['src_from_address'].unique()
rec_addresses = new_malicious['recipient'].unique()
all_addresses = list(set(src_addresses) | set(rec_addresses))
print(f"\nUnique addresses to fetch: {len(all_addresses)}")

# Load address model
model = joblib.load('machineLearning/models/random_forest_address_transfer_model_eth_3.joblib')
feature_names = list(model.feature_names_in_)
print(f"Loaded model with {len(feature_names)} features")


def collect_transfer_data(transfer, total_value, count_tx_over_1_eth, count_tx_over_10_eth, 
                          min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp):
    raw_ts = transfer.get('metadata', {}).get('blockTimestamp')
    if raw_ts:
        if earliest_timestamp is None or raw_ts < earliest_timestamp:
            earliest_timestamp = raw_ts
        if latest_timestamp is None or raw_ts > latest_timestamp:
            latest_timestamp = raw_ts

    value = transfer.get('value', 0) or 0
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


def compute_address_features(address: str, chain: str = "ethereum"):
    """Extract on-chain features for a single address"""
    try:
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
            unique_to_addresses.add(transfer.get('to', ''))

        for transfer in to_transfers['transfers']:
            total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp = collect_transfer_data(
                transfer, total_value, count_tx_over_1_eth, count_tx_over_10_eth, min_tx_val, max_tx_val, earliest_timestamp, latest_timestamp
            )
            unique_from_addresses.add(transfer.get('from', ''))

        # Convert timestamps
        if earliest_timestamp:
            earliest_timestamp = int(datetime.fromisoformat(earliest_timestamp.replace("Z", "+00:00")).timestamp() * 1000)
        else:
            earliest_timestamp = 0
        if latest_timestamp:
            latest_timestamp = int(datetime.fromisoformat(latest_timestamp.replace("Z", "+00:00")).timestamp() * 1000)
        else:
            latest_timestamp = 0

        if min_tx_val == float('inf'):
            min_tx_val = 0

        avg_value = total_value / total_transfers if total_transfers > 0 else 0
        time_active = latest_timestamp - earliest_timestamp
        avg_time_between_transactions = time_active / total_transfers if total_transfers > 0 else 0

        return {
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
            "avg_tx_freq": (total_transfers / (time_active / 86400000)) if time_active > 0 else 0,
        }
    except Exception as e:
        print(f"  Error fetching {address[:10]}...: {e}")
        return None


def predict_proba(address, address_features, model, feature_names):
    """Get 5-class probabilities for an address"""
    features = address_features.get(address)
    if features is None:
        return [1.0, 0.0, 0.0, 0.0, 0.0]  # Default to non-malicious
    
    feature_vector = [features.get(fname, 0) for fname in feature_names]
    X = np.array([feature_vector])
    return model.predict_proba(X)[0].tolist()


# Load address features from progress file (skip fetching)
print("\n" + "=" * 60)
print("Loading address features from progress file...")
print("=" * 60)

address_features = {}
if os.path.exists(PROGRESS_FILE):
    progress_df = pd.read_csv(PROGRESS_FILE)
    for _, row in progress_df.iterrows():
        addr = row['address']
        if row['success']:
            address_features[addr] = {fname: row[fname] for fname in feature_names}
        else:
            address_features[addr] = None
    print(f"Loaded {len(address_features)} addresses from progress file")
else:
    print("ERROR: No progress file found!")
    exit(1)

# Check how many addresses we're missing
missing = [a for a in all_addresses if a not in address_features]
print(f"Missing addresses (will use default probs): {len(missing)}")

success = sum(1 for v in address_features.values() if v is not None)
print(f"Successfully loaded features for {success}/{len(all_addresses)} addresses")

# Build enriched rows
print("\n" + "=" * 60)
print("Predicting probabilities and building rows...")
print("=" * 60)

max_source_index = enriched['source_index'].max()

enriched_rows = []
for i, (idx, row) in enumerate(new_malicious.iterrows()):
    src_probs = predict_proba(row['src_from_address'], address_features, model, feature_names)
    rec_probs = predict_proba(row['recipient'], address_features, model, feature_names)

    new_row = {
        'src_blockchain': row.get('src_blockchain', ''),
        'src_from_address': row['src_from_address'],
        'src_fee': row.get('src_fee', 0),
        'src_fee_usd': row.get('src_fee_usd', 0),
        'src_timestamp': row.get('src_timestamp', 0),
        'dst_blockchain': row.get('dst_blockchain', ''),
        'dst_fee': row.get('dst_fee', 0),
        'dst_fee_usd': row.get('dst_fee_usd', 0),
        'dst_timestamp': row.get('dst_timestamp', 0),
        'input_amount': row.get('input_amount', 0),
        'input_amount_usd': row.get('input_amount_usd', 0),
        'output_amount': row.get('output_amount', 0),
        'output_amount_usd': row.get('output_amount_usd', 0),
        'recipient': row['recipient'],
        'source_index': max_source_index + i + 1,
        'src_prob_class_0': src_probs[0],
        'src_prob_class_1': src_probs[1],
        'src_prob_class_2': src_probs[2],
        'src_prob_class_3': src_probs[3],
        'src_prob_class_4': src_probs[4],
        'rec_prob_class_0': rec_probs[0],
        'rec_prob_class_1': rec_probs[1],
        'rec_prob_class_2': rec_probs[2],
        'rec_prob_class_3': rec_probs[3],
        'rec_prob_class_4': rec_probs[4],
        'label': 1,
    }
    enriched_rows.append(new_row)

new_enriched = pd.DataFrame(enriched_rows)
print(f"Built {len(new_enriched)} new rows")

# Combine and save
combined = pd.concat([enriched, new_enriched], ignore_index=True)
print(f"\nCombined: {len(combined)} rows")
print(f"  Labels: {combined['label'].value_counts().sort_index().to_dict()}")

output_path = 'features/datasets/cross_chain_labeled_transactions_enriched_probs_v3.csv'
combined.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
