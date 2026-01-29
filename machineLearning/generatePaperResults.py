

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.utils import compute_sample_weight, class_weight
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from scipy import stats
import json
import os
from datetime import datetime

RESULTS_DIR = 'machineLearning/results'
N_FOLDS = 5
SEED = 40
TEST_SIZE = 0.33


ADDRESS_CLASS_NAMES = ['Non-Malicious', 'Phishing', 'Exploit', 'Sanctioned', 'Tornado']
TRANSACTION_CLASS_NAMES = ['Non-Malicious', 'Malicious']


def create_model_dirs(model_name):
    model_dir = f'{RESULTS_DIR}/{model_name}'
    figures_dir = f'{model_dir}/figures'
    tables_dir = f'{model_dir}/tables'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return model_dir, figures_dir, tables_dir


def get_per_class_metrics(y_true, y_pred, class_names, y_proba=None):
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    per_class = {}
    n_classes = len(class_names)
    
    per_class_auc = {}
    macro_auc = None
    if y_proba is not None:
        try:
            if n_classes == 2:
                macro_auc = roc_auc_score(y_true, y_proba[:, 1])
                per_class_auc[class_names[0]] = macro_auc
                per_class_auc[class_names[1]] = macro_auc
            else:
                y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                for i, name in enumerate(class_names):
                    if y_true_bin[:, i].sum() > 0:
                        per_class_auc[name] = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                macro_auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")
    
    for i, name in enumerate(class_names):
        if name in report:
            per_class[name] = {
                'precision': report[name]['precision'],
                'recall': report[name]['recall'],
                'f1': report[name]['f1-score'],
                'support': report[name]['support'],
                'auc': per_class_auc.get(name, None)
            }
    
    per_class['macro_avg'] = {
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1': report['macro avg']['f1-score'],
        'auc': macro_auc
    }
    per_class['weighted_avg'] = {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }
    
    return per_class


def print_per_class_results(per_class, class_names, title=""):
    if title:
        print(f"\n{title}")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Support':<10}")
    print("-" * 78)
    for name in class_names:
        if name in per_class:
            m = per_class[name]
            auc_str = f"{m['auc']*100:>10.2f}%" if m.get('auc') is not None else "       N/A"
            print(f"{name:<20} {m['precision']*100:>10.2f}% {m['recall']*100:>10.2f}% {m['f1']*100:>10.2f}% {auc_str} {m['support']:>8}")
    print("-" * 78)
    m = per_class['macro_avg']
    auc_str = f"{m['auc']*100:>10.2f}%" if m.get('auc') is not None else "       N/A"
    print(f"{'Macro Avg':<20} {m['precision']*100:>10.2f}% {m['recall']*100:>10.2f}% {m['f1']*100:>10.2f}% {auc_str}")
    m = per_class['weighted_avg']
    print(f"{'Weighted Avg':<20} {m['precision']*100:>10.2f}% {m['recall']*100:>10.2f}% {m['f1']*100:>10.2f}%")


def save_per_class_csv(per_class, class_names, filepath):
    rows = []
    for name in class_names:
        if name in per_class:
            m = per_class[name]
            rows.append({
                'class': name,
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'auc': m.get('auc'),
                'support': m['support']
            })
    rows.append({'class': 'macro_avg', **per_class['macro_avg']})
    rows.append({'class': 'weighted_avg', **per_class['weighted_avg']})
    pd.DataFrame(rows).to_csv(filepath, index=False)


def plot_confusion_matrix(y_true, y_pred, class_names, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{filepath}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filepath}.pdf', bbox_inches='tight')
    plt.close()

def load_address_dataset():
    dataset = pd.read_csv('features/datasets/address_transfer_features_eth_3.csv')
    X = dataset.drop(columns=['label', 'address'])
    y = dataset['label'].values
    return X, y


def get_address_models():
    def train_rf(X_train, y_train, sample_weights=None):
        model = RandomForestClassifier(
            n_estimators=900, max_depth=None, min_samples_split=3,
            min_samples_leaf=2, max_features='log2', bootstrap=False,
            n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model
    
    def train_lr(X_train, y_train, sample_weights=None):
        logreg = LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs',
            max_iter=1000, multi_class='multinomial', random_state=42
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('logreg', logreg)])
        if sample_weights is not None:
            pipe.fit(X_train, y_train, logreg__sample_weight=sample_weights)
        else:
            pipe.fit(X_train, y_train)
        return pipe
    
    def train_xgb(X_train, y_train, sample_weights=None):
        model = xgb.XGBClassifier(
            objective='multi:softprob', max_depth=12, learning_rate=0.075,
            n_estimators=1200, subsample=1, reg_lambda=1, reg_alpha=0,
            min_child_weight=7, gamma=0, colsample_bytree=1.0
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model
    
    return {'RandomForest': train_rf, 'LogisticRegression': train_lr, 'XGBoost': train_xgb}


def run_model1_address_classification():
    print("\n" + "=" * 80)
    print("MODEL 1: ADDRESS CLASSIFICATION (Multi-class, 5 classes)")
    print("Dataset: features/datasets/address_transfer_features_eth_3.csv")
    print("=" * 80)
    
    model_dir, figures_dir, tables_dir = create_model_dirs('model1_address')
    X, y = load_address_dataset()
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(ADDRESS_CLASS_NAMES, np.bincount(y)))}")
    
    majority_class = np.bincount(y).argmax()
    baseline_acc = np.mean(y == majority_class)
    print(f"Majority baseline accuracy: {baseline_acc:.4f}")
    
    models = get_address_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    results = {}
    
    for model_name, train_func in models.items():
        print(f"\n--- {model_name} ---")
        
        all_y_true, all_y_pred, all_y_proba = [], [], []
        fold_accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            classes = np.unique(y_train)
            weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([dict(zip(classes, weights))[label] for label in y_train])
            
            model = train_func(X_train, y_train, sample_weights)
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                all_y_proba.append(y_proba)
            
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.vstack(all_y_proba) if all_y_proba else None
        
        per_class = get_per_class_metrics(all_y_true, all_y_pred, ADDRESS_CLASS_NAMES, all_y_proba)
        accuracy = accuracy_score(all_y_true, all_y_pred)
        
        print(f"Accuracy: {accuracy*100:.2f}% (std: {np.std(fold_accuracies)*100:.2f}%)")
        print_per_class_results(per_class, ADDRESS_CLASS_NAMES)
        
        results[model_name] = {
            'accuracy': {'mean': float(accuracy), 'std': float(np.std(fold_accuracies))},
            'per_class': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv 
                             for kk, vv in v.items()} for k, v in per_class.items()}
        }
        
        save_per_class_csv(per_class, ADDRESS_CLASS_NAMES, f'{tables_dir}/{model_name}_per_class.csv')
        
        plot_confusion_matrix(all_y_true, all_y_pred, ADDRESS_CLASS_NAMES,
                            f'Model 1: Address Classification - {model_name}',
                            f'{figures_dir}/{model_name}_confusion_matrix')
    
    output = {
        'model': 'Model 1: Address Classification',
        'dataset': 'features/datasets/address_transfer_features_eth_3.csv',
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1]),
        'class_names': ADDRESS_CLASS_NAMES,
        'class_distribution': {name: int(count) for name, count in zip(ADDRESS_CLASS_NAMES, np.bincount(y))},
        'baseline_accuracy': float(baseline_acc),
        'classifiers': results
    }
    
    with open(f'{model_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {model_dir}/")
    return output

def load_transaction_dataset():
    """Load transaction dataset"""
    dataset = pd.read_csv('data/datasets/labeled_cross_chain_transactions_3.csv', low_memory=False)
    
    malicious = dataset[dataset['label'] > 0]
    non_malicious = dataset[dataset['label'] == 0].sample(n=50000, random_state=SEED)
    dataset = pd.concat([malicious, non_malicious], ignore_index=True).sample(frac=1.0, random_state=SEED)
    
    dataset['label'] = (dataset['label'] > 0).astype(int)
    
    feature_cols = ['src_fee', 'src_fee_usd', 'dst_fee', 'dst_fee_usd',
                    'input_amount', 'input_amount_usd', 'output_amount', 'output_amount_usd']
    X = dataset[feature_cols].fillna(0)
    y = dataset['label'].values
    
    return X, y


def get_transaction_models():
    def train_rf(X_train, y_train, sample_weights=None):
        model = RandomForestClassifier(
            n_estimators=900, max_depth=None, min_samples_split=3,
            min_samples_leaf=2, max_features='log2', bootstrap=False,
            n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model
    
    def train_lr(X_train, y_train, sample_weights=None):
        logreg = LogisticRegression(
            penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42
        )
        pipe = Pipeline([('scaler', StandardScaler()), ('logreg', logreg)])
        if sample_weights is not None:
            pipe.fit(X_train, y_train, logreg__sample_weight=sample_weights)
        else:
            pipe.fit(X_train, y_train)
        return pipe
    
    def train_xgb(X_train, y_train, sample_weights=None):
        model = xgb.XGBClassifier(
            objective='binary:logistic', max_depth=12, learning_rate=0.075,
            n_estimators=1200, subsample=1, reg_lambda=1, reg_alpha=0,
            min_child_weight=7, gamma=0, colsample_bytree=1.0
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model
    
    return {'RandomForest': train_rf, 'LogisticRegression': train_lr, 'XGBoost': train_xgb}


def run_model2_transaction_classification():
    print("\n" + "=" * 80)
    print("MODEL 2: TRANSACTION CLASSIFICATION (Binary)")
    print("Dataset: data/datasets/labeled_cross_chain_transactions_3.csv")
    print("=" * 80)
    
    model_dir, figures_dir, tables_dir = create_model_dirs('model2_transaction')
    X, y = load_transaction_dataset()
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(TRANSACTION_CLASS_NAMES, np.bincount(y)))}")
    
    majority_class = np.bincount(y).argmax()
    baseline_acc = np.mean(y == majority_class)
    print(f"Majority baseline accuracy: {baseline_acc:.4f}")
    
    models = get_transaction_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    results = {}
    
    for model_name, train_func in models.items():
        print(f"\n--- {model_name} ---")
        
        all_y_true, all_y_pred, all_y_proba = [], [], []
        fold_accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            classes = np.unique(y_train)
            weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.array([dict(zip(classes, weights))[label] for label in y_train])
            
            model = train_func(X_train, y_train, sample_weights)
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                all_y_proba.append(y_proba)
            
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_proba = np.vstack(all_y_proba) if all_y_proba else None
        
        per_class = get_per_class_metrics(all_y_true, all_y_pred, TRANSACTION_CLASS_NAMES, all_y_proba)
        accuracy = accuracy_score(all_y_true, all_y_pred)
        
        print(f"Accuracy: {accuracy*100:.2f}% (std: {np.std(fold_accuracies)*100:.2f}%)")
        print_per_class_results(per_class, TRANSACTION_CLASS_NAMES)
        

        results[model_name] = {
            'accuracy': {'mean': float(accuracy), 'std': float(np.std(fold_accuracies))},
            'per_class': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv 
                             for kk, vv in v.items()} for k, v in per_class.items()}
        }
        

        save_per_class_csv(per_class, TRANSACTION_CLASS_NAMES, f'{tables_dir}/{model_name}_per_class.csv')
        

        plot_confusion_matrix(all_y_true, all_y_pred, TRANSACTION_CLASS_NAMES,
                            f'Model 2: Transaction Classification - {model_name}',
                            f'{figures_dir}/{model_name}_confusion_matrix')
    

    output = {
        'model': 'Model 2: Transaction Classification',
        'dataset': 'data/datasets/labeled_cross_chain_transactions_3.csv',
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1]),
        'class_names': TRANSACTION_CLASS_NAMES,
        'class_distribution': {name: int(count) for name, count in zip(TRANSACTION_CLASS_NAMES, np.bincount(y))},
        'baseline_accuracy': float(baseline_acc),
        'classifiers': results
    }
    
    with open(f'{model_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {model_dir}/")
    return output



def load_hybrid_dataset():
    dataset = pd.read_csv('features/datasets/cross_chain_labeled_transactions_enriched_probs_v3.csv')
    
    src_prob_cols = ['src_prob_class_0', 'src_prob_class_1', 'src_prob_class_2', 'src_prob_class_3', 'src_prob_class_4']
    rec_prob_cols = ['rec_prob_class_0', 'rec_prob_class_1', 'rec_prob_class_2', 'rec_prob_class_3', 'rec_prob_class_4']
    
    src_probs = dataset[src_prob_cols].values
    rec_probs = dataset[rec_prob_cols].values
    
    y_src = src_probs.argmax(axis=1)
    y_rec = rec_probs.argmax(axis=1)
    y = np.column_stack([y_src, y_rec])
    
    drop_cols = ['label', 'source_index', 'src_from_address', 'recipient', 'src_blockchain', 'dst_blockchain'] + src_prob_cols + rec_prob_cols
    X = dataset.drop(columns=drop_cols)
    
    return X, y


def get_hybrid_models():
    def train_rf(X_train, y_train):
        base_rf = RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1,
            class_weight='balanced_subsample', max_depth=15, min_samples_leaf=2
        )
        model = MultiOutputClassifier(base_rf, n_jobs=-1)
        model.fit(X_train, y_train)
        return model
    
    def train_lr(X_train, y_train):
        base_lr = LogisticRegression(
            max_iter=2000, random_state=SEED, class_weight='balanced',
            solver='saga', C=0.5
        )
        model = MultiOutputClassifier(base_lr, n_jobs=-1)
        model.fit(X_train, y_train)
        return model
    
    def train_xgb(X_train, y_train):
        weights_src = compute_sample_weight('balanced', y_train[:, 0])
        weights_rec = compute_sample_weight('balanced', y_train[:, 1])
        sample_weights = np.sqrt(weights_src * weights_rec)
        
        base_xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            random_state=SEED, eval_metric='mlogloss',
            reg_alpha=0.1, reg_lambda=1.0
        )
        model = MultiOutputClassifier(base_xgb, n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model
    
    return {'RandomForest': train_rf, 'LogisticRegression': train_lr, 'XGBoost': train_xgb}


def run_model3_hybrid_classification():
    print("\n" + "=" * 80)
    print("MODEL 3: HYBRID MULTI-OUTPUT CLASSIFICATION")
    print("Dataset: features/datasets/cross_chain_labeled_transactions_enriched_probs_v3.csv")
    print("=" * 80)
    
    model_dir, figures_dir, tables_dir = create_model_dirs('model3_hybrid')
    X, y = load_hybrid_dataset()
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Source class distribution: {dict(zip(ADDRESS_CLASS_NAMES, np.bincount(y[:, 0])))}")
    print(f"Recipient class distribution: {dict(zip(ADDRESS_CLASS_NAMES, np.bincount(y[:, 1])))}")
  
    src_majority = np.bincount(y[:, 0]).argmax()
    rec_majority = np.bincount(y[:, 1]).argmax()
    src_baseline = np.mean(y[:, 0] == src_majority)
    rec_baseline = np.mean(y[:, 1] == rec_majority)
    print(f"Source baseline accuracy: {src_baseline:.4f}")
    print(f"Recipient baseline accuracy: {rec_baseline:.4f}")
    
    models = get_hybrid_models()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    results = {}
    
    for model_name, train_func in models.items():
        print(f"\n--- {model_name} ---")
        
        all_src_true, all_src_pred, all_src_proba = [], [], []
        all_rec_true, all_rec_pred, all_rec_proba = [], [], []
        fold_src_acc, fold_rec_acc, fold_exact = [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y[:, 0]), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = train_func(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if hasattr(model, 'estimators_') and len(model.estimators_) >= 2:
                if hasattr(model.estimators_[0], 'predict_proba'):
                    src_proba = model.estimators_[0].predict_proba(X_test)
                    rec_proba = model.estimators_[1].predict_proba(X_test)
                    all_src_proba.append(src_proba)
                    all_rec_proba.append(rec_proba)
            
            fold_src_acc.append(accuracy_score(y_test[:, 0], y_pred[:, 0]))
            fold_rec_acc.append(accuracy_score(y_test[:, 1], y_pred[:, 1]))
            fold_exact.append(np.mean(np.all(y_pred == y_test, axis=1)))
            
            all_src_true.extend(y_test[:, 0])
            all_src_pred.extend(y_pred[:, 0])
            all_rec_true.extend(y_test[:, 1])
            all_rec_pred.extend(y_pred[:, 1])
        
        all_src_true = np.array(all_src_true)
        all_src_pred = np.array(all_src_pred)
        all_rec_true = np.array(all_rec_true)
        all_rec_pred = np.array(all_rec_pred)
        all_src_proba = np.vstack(all_src_proba) if all_src_proba else None
        all_rec_proba = np.vstack(all_rec_proba) if all_rec_proba else None
        
        src_per_class = get_per_class_metrics(all_src_true, all_src_pred, ADDRESS_CLASS_NAMES, all_src_proba)
        rec_per_class = get_per_class_metrics(all_rec_true, all_rec_pred, ADDRESS_CLASS_NAMES, all_rec_proba)
        
        src_acc = accuracy_score(all_src_true, all_src_pred)
        rec_acc = accuracy_score(all_rec_true, all_rec_pred)
        exact_match = np.mean(fold_exact)
        
        print(f"Source Accuracy: {src_acc*100:.2f}% | Recipient Accuracy: {rec_acc*100:.2f}% | Exact Match: {exact_match*100:.2f}%")
        print_per_class_results(src_per_class, ADDRESS_CLASS_NAMES, "Source Address:")
        print_per_class_results(rec_per_class, ADDRESS_CLASS_NAMES, "Recipient Address:")
        
        results[model_name] = {
            'source_accuracy': {'mean': float(src_acc), 'std': float(np.std(fold_src_acc))},
            'recipient_accuracy': {'mean': float(rec_acc), 'std': float(np.std(fold_rec_acc))},
            'exact_match': {'mean': float(exact_match), 'std': float(np.std(fold_exact))},
            'source_per_class': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv 
                                    for kk, vv in v.items()} for k, v in src_per_class.items()},
            'recipient_per_class': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv 
                                       for kk, vv in v.items()} for k, v in rec_per_class.items()}
        }
        
        save_per_class_csv(src_per_class, ADDRESS_CLASS_NAMES, f'{tables_dir}/{model_name}_source_per_class.csv')
        save_per_class_csv(rec_per_class, ADDRESS_CLASS_NAMES, f'{tables_dir}/{model_name}_recipient_per_class.csv')
        
        plot_confusion_matrix(all_src_true, all_src_pred, ADDRESS_CLASS_NAMES,
                            f'Model 3: Hybrid - {model_name} (Source)',
                            f'{figures_dir}/{model_name}_source_confusion_matrix')
        plot_confusion_matrix(all_rec_true, all_rec_pred, ADDRESS_CLASS_NAMES,
                            f'Model 3: Hybrid - {model_name} (Recipient)',
                            f'{figures_dir}/{model_name}_recipient_confusion_matrix')
    
    output = {
        'model': 'Model 3: Hybrid Multi-Output Classification',
        'dataset': 'features/datasets/cross_chain_labeled_transactions_enriched_probs_v3.csv',
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1]),
        'class_names': ADDRESS_CLASS_NAMES,
        'source_class_distribution': {name: int(count) for name, count in zip(ADDRESS_CLASS_NAMES, np.bincount(y[:, 0]))},
        'recipient_class_distribution': {name: int(count) for name, count in zip(ADDRESS_CLASS_NAMES, np.bincount(y[:, 1]))},
        'source_baseline_accuracy': float(src_baseline),
        'recipient_baseline_accuracy': float(rec_baseline),
        'classifiers': results
    }
    
    with open(f'{model_dir}/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {model_dir}/")
    return output


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\n{'='*80}")
    print(f"PAPER RESULTS GENERATOR - XChainAnomaly")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*80}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    model1_results = run_model1_address_classification()
    model2_results = run_model2_transaction_classification()
    model3_results = run_model3_hybrid_classification()
    
    combined = {
        'timestamp': timestamp,
        'config': {'n_folds': N_FOLDS, 'seed': SEED, 'test_size': TEST_SIZE},
        'model1_address': model1_results,
        'model2_transaction': model2_results,
        'model3_hybrid': model3_results
    }
    
    with open(f'{RESULTS_DIR}/all_models_results.json', 'w') as f:
        json.dump(combined, f, indent=2)
    


if __name__ == "__main__":
    main()
