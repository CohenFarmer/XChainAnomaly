import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import compute_sample_weight
import xgboost as xgb

dataset = pd.read_csv('features/datasets/cross_chain_labeled_transactions_enriched_probs.csv')

test_size = 0.33
seed = 40

src_prob_cols = ['src_prob_class_0', 'src_prob_class_1', 'src_prob_class_2', 'src_prob_class_3', 'src_prob_class_4']
rec_prob_cols = ['rec_prob_class_0', 'rec_prob_class_1', 'rec_prob_class_2', 'rec_prob_class_3', 'rec_prob_class_4']

y_src = dataset[src_prob_cols].values.argmax(axis=1)
y_rec = dataset[rec_prob_cols].values.argmax(axis=1)
y = np.column_stack([y_src, y_rec])

drop_cols = ['label', 'source_index', 'src_from_address', 'recipient', 'src_blockchain', 'dst_blockchain'] + src_prob_cols + rec_prob_cols
X = dataset.drop(columns=drop_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed, stratify=y_src
)


def compute_xgb_sample_weights(y_train):
    weights_src = compute_sample_weight('balanced', y_train[:, 0])
    weights_rec = compute_sample_weight('balanced', y_train[:, 1])
    combined_weights = np.sqrt(weights_src * weights_rec)
    return combined_weights


def train_multi_output_random_forest(X_train, y_train):
    base_rf = RandomForestClassifier(
        n_estimators=200, 
        random_state=seed, 
        n_jobs=-1, 
        class_weight='balanced_subsample',
        max_depth=15,
        min_samples_leaf=2
    )
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_multi_output_logistic_regression(X_train, y_train):
    base_lr = LogisticRegression(
        max_iter=2000, 
        random_state=seed, 
        class_weight='balanced',
        solver='saga',
        C=0.5
    )
    model = MultiOutputClassifier(base_lr, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_multi_output_xgboost(X_train, y_train):
    sample_weights = compute_xgb_sample_weights(y_train)
    
    base_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        random_state=seed,
        eval_metric='mlogloss',
        scale_pos_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    model = MultiOutputClassifier(base_xgb, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def test_multi_output_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    
    print("\n--- Source Address Classification ---")
    print(f"Accuracy: {accuracy_score(y_test[:, 0], y_pred[:, 0]):.4f}")
    print(classification_report(y_test[:, 0], y_pred[:, 0], zero_division=0))
    
    print("\n--- Recipient Address Classification ---")
    print(f"Accuracy: {accuracy_score(y_test[:, 1], y_pred[:, 1]):.4f}")
    print(classification_report(y_test[:, 1], y_pred[:, 1], zero_division=0))
    
    exact_match = np.mean(np.all(y_pred == y_test, axis=1))
    print(f"\nExact Match Ratio (both outputs correct): {exact_match:.4f}")
    
    return y_pred


if __name__ == "__main__":
    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"\nSource class distribution: {np.bincount(y_src)}")
    print(f"Recipient class distribution: {np.bincount(y_rec)}")
    
    print("\nTraining Random Forest Multi-Output...")
    rf_model = train_multi_output_random_forest(X_train, y_train)
    test_multi_output_model(rf_model, X_test, y_test, "Random Forest")
    
    print("\nTraining Logistic Regression Multi-Output...")
    lr_model = train_multi_output_logistic_regression(X_train, y_train)
    test_multi_output_model(lr_model, X_test, y_test, "Logistic Regression")
    
    print("\nTraining XGBoost Multi-Output...")
    xgb_model = train_multi_output_xgboost(X_train, y_train)
    test_multi_output_model(xgb_model, X_test, y_test, "XGBoost")
