import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from xgboost.callback import EarlyStopping
from sklearn.metrics import make_scorer, f1_score
from machineLearning.supervisedModels import train_xgboost_model, test_model, train_logistic_regression_model, train_random_forest_model

# Load labeled transaction data
dataset = pd.read_csv('data/datasets/labeled_cross_chain_transactions_3.csv', low_memory=False)

# Balance the dataset: all malicious (label > 0) + 50k non-malicious (label 0)
malicious = dataset[dataset['label'] > 0]
non_malicious = dataset[dataset['label'] == 0].sample(n=50000, random_state=40)
dataset = pd.concat([malicious, non_malicious], ignore_index=True).sample(frac=1.0, random_state=40)

# Convert labels to binary: 0 = non-malicious, 1 = malicious
dataset['label'] = (dataset['label'] > 0).astype(int)

print(f"Dataset size: {len(dataset)}")
print(f"Malicious: {len(malicious)}, Non-malicious: {len(non_malicious)}")
print(f"Label distribution:\n{dataset['label'].value_counts()}")

test_size = 0.33
seed = 40

src_blockchain_encoder = LabelEncoder()
dst_blockchain_encoder = LabelEncoder()
dataset['src_blockchain_encoded'] = src_blockchain_encoder.fit_transform(dataset['src_blockchain'].fillna('unknown'))
dataset['dst_blockchain_encoded'] = dst_blockchain_encoder.fit_transform(dataset['dst_blockchain'].fillna('unknown'))

feature_cols = ['src_fee', 'src_fee_usd', 'dst_fee', 'dst_fee_usd', 
                'input_amount', 'input_amount_usd', 'output_amount', 'output_amount_usd',
                'src_blockchain_encoded', 'dst_blockchain_encoded']

X = dataset[feature_cols].fillna(0)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

classes = np.unique(y_train)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_to_weight = {c: w for c, w in zip(classes, class_weights)}
sample_weights = np.array([class_to_weight[label] for label in y_train])

random_forest = train_random_forest_model(X_train, y_train, sample_weights=sample_weights, binary=True)
logistic_r = train_logistic_regression_model(X_train, y_train, sample_weights=sample_weights, binary=True)
xgboost_model = train_xgboost_model(X_train, y_train, sample_weights=sample_weights, binary=True)

test_model(random_forest, X_test, y_test, binary=True)
test_model(logistic_r, X_test, y_test, binary=True)
test_model(xgboost_model, X_test, y_test, binary=True)

