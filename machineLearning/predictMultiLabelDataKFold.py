import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

dataset = pd.read_csv('features/datasets/cross_chain_labeled_transactions_enriched_probs.csv')

n_folds = 5
seed = 40

src_prob_cols = ['src_prob_class_0', 'src_prob_class_1', 'src_prob_class_2', 'src_prob_class_3', 'src_prob_class_4']
rec_prob_cols = ['rec_prob_class_0', 'rec_prob_class_1', 'rec_prob_class_2', 'rec_prob_class_3', 'rec_prob_class_4']

y_src = dataset[src_prob_cols].values.argmax(axis=1)
y_rec = dataset[rec_prob_cols].values.argmax(axis=1)

# Remap labels to consecutive integers for XGBoost compatibility
# Original: 0=Non-malicious, 1=Phishing, 2=Exploit, 3=Sanctioned, 4=Tornado
# Store original labels for reporting
src_label_encoder = LabelEncoder()
rec_label_encoder = LabelEncoder()
y_src_encoded = src_label_encoder.fit_transform(y_src)
y_rec_encoded = rec_label_encoder.fit_transform(y_rec)

print(f"Source label mapping: {dict(zip(src_label_encoder.classes_, range(len(src_label_encoder.classes_))))}")
print(f"Recipient label mapping: {dict(zip(rec_label_encoder.classes_, range(len(rec_label_encoder.classes_))))}")

y = np.column_stack([y_src_encoded, y_rec_encoded])
# Keep original labels for reporting
y_original = np.column_stack([y_src, y_rec])

drop_cols = ['label', 'source_index', 'src_from_address', 'recipient', 'src_blockchain', 'dst_blockchain'] + src_prob_cols + rec_prob_cols
X = dataset.drop(columns=drop_cols)


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


def evaluate_fold(y_true, y_pred):
    src_acc = accuracy_score(y_true[:, 0], y_pred[:, 0])
    rec_acc = accuracy_score(y_true[:, 1], y_pred[:, 1])
    exact_match = np.mean(np.all(y_pred == y_true, axis=1))
    return src_acc, rec_acc, exact_match


def run_kfold_evaluation(X, y, model_name, train_func):
    print(f"\n{'='*60}")
    print(f"{model_name} - {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    src_accuracies = []
    rec_accuracies = []
    exact_matches = []
    
    all_y_true_src = []
    all_y_pred_src = []
    all_y_true_rec = []
    all_y_pred_rec = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y[:, 0]), 1):
        print(f"\nFold {fold}/{n_folds}...")
        print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
        model = train_func(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        src_acc, rec_acc, exact_match = evaluate_fold(y_test, y_pred)
        src_accuracies.append(src_acc)
        rec_accuracies.append(rec_acc)
        exact_matches.append(exact_match)
        
        all_y_true_src.extend(y_test[:, 0])
        all_y_pred_src.extend(y_pred[:, 0])
        all_y_true_rec.extend(y_test[:, 1])
        all_y_pred_rec.extend(y_pred[:, 1])
        
        print(f"  Source Accuracy: {src_acc:.4f}")
        print(f"  Recipient Accuracy: {rec_acc:.4f}")
        print(f"  Exact Match: {exact_match:.4f}")
    
    print(f"\n{'-'*60}")
    print(f"{model_name} - Summary (Mean ± Std)")
    print(f"{'-'*60}")
    print(f"Source Accuracy:    {np.mean(src_accuracies):.4f} ± {np.std(src_accuracies):.4f}")
    print(f"Recipient Accuracy: {np.mean(rec_accuracies):.4f} ± {np.std(rec_accuracies):.4f}")
    print(f"Exact Match Ratio:  {np.mean(exact_matches):.4f} ± {np.std(exact_matches):.4f}")
    
    print(f"\n{'-'*60}")
    print(f"{model_name} - Overall Classification Reports (All Folds Combined)")
    print(f"{'-'*60}")
    
    print("\n--- Source Address Classification ---")
    print(classification_report(all_y_true_src, all_y_pred_src, zero_division=0))
    
    print("\n--- Recipient Address Classification ---")
    print(classification_report(all_y_true_rec, all_y_pred_rec, zero_division=0))
    
    return {
        'src_acc_mean': np.mean(src_accuracies),
        'src_acc_std': np.std(src_accuracies),
        'rec_acc_mean': np.mean(rec_accuracies),
        'rec_acc_std': np.std(rec_accuracies),
        'exact_match_mean': np.mean(exact_matches),
        'exact_match_std': np.std(exact_matches)
    }


if __name__ == "__main__":
    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Total samples: {X.shape[0]}")
    print(f"Number of folds: {n_folds}")
    print(f"\nSource class distribution: {np.bincount(y_src)}")
    print(f"Recipient class distribution: {np.bincount(y_rec)}")
    
    results = {}
    
    print("\n" + "="*60)
    print("Running K-Fold Cross-Validation")
    print("="*60)
    
    # results['Random Forest'] = run_kfold_evaluation(
    #     X, y, "Random Forest", train_multi_output_random_forest
    # )
    
    results['XGBoost'] = run_kfold_evaluation(
        X, y, "XGBoost", train_multi_output_xgboost
    )
    
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<25} {'Src Acc':<15} {'Rec Acc':<15} {'Exact Match':<15}")
    print("-"*70)
    for model_name, metrics in results.items():
        src = f"{metrics['src_acc_mean']:.4f}±{metrics['src_acc_std']:.4f}"
        rec = f"{metrics['rec_acc_mean']:.4f}±{metrics['rec_acc_std']:.4f}"
        exact = f"{metrics['exact_match_mean']:.4f}±{metrics['exact_match_std']:.4f}"
        print(f"{model_name:<25} {src:<15} {rec:<15} {exact:<15}")