import pandas as pd
import numpy as np
import joblib

cctx_tx_data = pd.read_csv('features/datasets/cross_chain_labeled_transactions_balanced_50k.csv')

cctx_tx_data = cctx_tx_data.reset_index(drop=True)
cctx_tx_data['source_index'] = np.arange(len(cctx_tx_data))

if 'source_index' not in cctx_tx_data.columns:
	cctx_tx_data = cctx_tx_data.copy()
	cctx_tx_data['source_index'] = cctx_tx_data.index
cctx_tx_features = pd.read_csv('features/datasets/cctx_transfer_features.csv')

cctx_src_from_tx_data = cctx_tx_features[cctx_tx_features['role'] == 'src_from']
cctx_recipient_tx_data = cctx_tx_features[cctx_tx_features['role'] == 'recipient']

cctx_src_from_tx_data = cctx_src_from_tx_data.sort_values(by=['source_index'])
cctx_recipient_tx_data = cctx_recipient_tx_data.sort_values(by=['source_index'])

print(cctx_src_from_tx_data.head(10))
print(cctx_recipient_tx_data.head(10))

print(len(cctx_src_from_tx_data), len(cctx_recipient_tx_data), len(cctx_tx_data))

rf_model = joblib.load('machineLearning/models/random_forest_address_transfer_model_eth_3.joblib')

src_indices = cctx_src_from_tx_data['source_index'].to_numpy()
rec_indices = cctx_recipient_tx_data['source_index'].to_numpy()

src_address = cctx_src_from_tx_data['address'].to_numpy()
rec_address = cctx_recipient_tx_data['address'].to_numpy()

X_src_from = cctx_src_from_tx_data.drop(columns=['address', 'label', 'role', 'source_index'])
X_recipient = cctx_recipient_tx_data.drop(columns=['address', 'label', 'role', 'source_index'])

y_pred_src_from = rf_model.predict_proba(X_src_from)
y_prec_recipient = rf_model.predict_proba(X_recipient)

results_src = pd.DataFrame(y_pred_src_from, columns=[f'src_prob_class_{i}' for i in range(y_pred_src_from.shape[1])])
results_src['source_index'] = src_indices
results_src['address'] = src_address

results_rec = pd.DataFrame(y_prec_recipient, columns=[f'rec_prob_class_{i}' for i in range(y_prec_recipient.shape[1])])
results_rec['source_index'] = rec_indices
results_rec['address'] = rec_address

print(results_src.head())
print(results_rec.head())

print(cctx_tx_data.columns)

print(cctx_tx_data.head())

src_probs = results_src.drop(columns=['address'])
rec_probs = results_rec.drop(columns=['address'])

enriched = (
	cctx_tx_data
		.merge(src_probs, on='source_index', how='left')
		.merge(rec_probs, on='source_index', how='left')
)

out_path = 'features/datasets/cross_chain_labeled_transactions_enriched_probs_rf.csv'
prob_cols = [c for c in enriched.columns if c.startswith('src_prob_class_') or c.startswith('rec_prob_class_')]
before_ct = len(enriched)
if prob_cols:
	enriched = enriched.dropna(subset=prob_cols)
	print('Dropped rows missing probabilities:', before_ct - len(enriched))
if 'label' in enriched.columns:
	ordered_cols = [c for c in enriched.columns if c != 'label'] + ['label']
	enriched = enriched[ordered_cols]
enriched.to_csv(out_path, index=False)
print('Wrote enriched dataset to', out_path)

