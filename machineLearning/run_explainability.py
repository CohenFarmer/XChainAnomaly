#demo script for shap and llm based transaction explanations

import numpy as np
import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight
import os

from machineLearning.explainability import TransactionExplainer, LLMExplainer, LocalLLMExplainer


SEED = 40
MODEL_PATH = 'machineLearning/models/multi_output_rf_model.joblib'
USE_LLM = True  
LLM_PROVIDER = "local"  

#loads dataset and prepares features and labels
def load_data():

    dataset = pd.read_csv('features/datasets/cross_chain_labeled_transactions_enriched_probs_v3.csv')
    
    src_prob_cols = ['src_prob_class_0', 'src_prob_class_1', 'src_prob_class_2', 'src_prob_class_3', 'src_prob_class_4']
    rec_prob_cols = ['rec_prob_class_0', 'rec_prob_class_1', 'rec_prob_class_2', 'rec_prob_class_3', 'rec_prob_class_4']
    

    y_src = dataset[src_prob_cols].values.argmax(axis=1)
    y_rec = dataset[rec_prob_cols].values.argmax(axis=1)
    y = np.column_stack([y_src, y_rec])
    

    drop_cols = ['label', 'source_index', 'src_from_address', 'recipient', 'src_blockchain', 'dst_blockchain'] + src_prob_cols + rec_prob_cols
    drop_cols = [c for c in drop_cols if c in dataset.columns]
    X = dataset.drop(columns=drop_cols)
    
    return X, y, dataset

#trains multi output random forest with balanced weights
def train_model(X_train, y_train):
    
    print("Training Model 3 (Multi-Output Random Forest)...")
    
  
    weights_src = compute_sample_weight('balanced', y_train[:, 0])
    weights_rec = compute_sample_weight('balanced', y_train[:, 1])
    sample_weights = np.sqrt(weights_src * weights_rec)
    
    base_rf = RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_depth=15,
        min_samples_leaf=2
    )
    
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    return model

#returns llm explainer based on configured provider
def get_llm_explainer():

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI() 
        return LLMExplainer(client, model_name="gpt-4o-mini")
    
    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic
        from machineLearning.explainability import AnthropicExplainer
        client = Anthropic() 
        return AnthropicExplainer(client)
    
    elif LLM_PROVIDER == "local":
        return LocalLLMExplainer(
            base_url="http://localhost:11434",
            model_name="gemma3:4b" 
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")

#runs explainability demo with interactive mode
def main():
    print("Loading data...")
    X, y, dataset = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=SEED, stratify=y[:, 0]
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features: {list(X.columns)}")
    

    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
    else:
        model = train_model(X_train, y_train)
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    print("\nInitializing explainer...")
    tx_explainer = TransactionExplainer(model, feature_names=X.columns.tolist())
    
    sample_idx = 0 
    sample = X_test.iloc[sample_idx]
    
    print("\n" + "="*60)
    print("TRANSACTION EXPLANATION")
    print("="*60)
    
    shap_result = tx_explainer.get_shap_explanation(sample)
    
    print(f"\nTransaction Features:")
    for name, value in shap_result['feature_values'].items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\n--- SOURCE ADDRESS ---")
    print(f"Prediction: {shap_result['source']['predicted_label']}")
    if shap_result['source']['probabilities']:
        print(f"Confidence: {shap_result['source']['probabilities'][shap_result['source']['predicted_label']]*100:.1f}%")
    print("Top contributing features:")
    for feat in shap_result['source']['feature_contributions'][:5]:
        sign = "+" if feat['contribution'] > 0 else ""
        print(f"  {feat['feature']}: {feat['value']:.4f} (contribution: {sign}{feat['contribution']:.4f})")
    
    print(f"\n--- RECIPIENT ADDRESS ---")
    print(f"Prediction: {shap_result['recipient']['predicted_label']}")
    if shap_result['recipient']['probabilities']:
        print(f"Confidence: {shap_result['recipient']['probabilities'][shap_result['recipient']['predicted_label']]*100:.1f}%")
    print("Top contributing features:")
    for feat in shap_result['recipient']['feature_contributions'][:5]:
        sign = "+" if feat['contribution'] > 0 else ""
        print(f"  {feat['feature']}: {feat['value']:.4f} (contribution: {sign}{feat['contribution']:.4f})")
    

    if USE_LLM:
        print("\n" + "="*60)
        print("LLM NATURAL LANGUAGE EXPLANATION")
        print("="*60)
        
        try:
            llm_explainer = get_llm_explainer()
            explanation = llm_explainer.explain(shap_result)
            print(explanation)
        except Exception as e:
            print(f"LLM explanation failed: {e}")
            print("Make sure your API key is set or Ollama is running.")
    

    print("\n" + "="*60)
    while True:
        user_input = input("\nEnter transaction index to explain (0-{}) or 'q' to quit: ".format(len(X_test)-1))
        if user_input.lower() == 'q':
            break
        try:
            idx = int(user_input)
            if 0 <= idx < len(X_test):
                sample = X_test.iloc[idx]
                shap_result = tx_explainer.get_shap_explanation(sample)
                
                print(f"\n--- SOURCE: {shap_result['source']['predicted_label']} ---")
                for feat in shap_result['source']['feature_contributions'][:3]:
                    print(f"  {feat['feature']}: {feat['value']:.4f}")
                
                print(f"\n--- RECIPIENT: {shap_result['recipient']['predicted_label']} ---")
                for feat in shap_result['recipient']['feature_contributions'][:3]:
                    print(f"  {feat['feature']}: {feat['value']:.4f}")
                
                if USE_LLM:
                    if input("\nGenerate LLM explanation? (y/n): ").lower() == 'y':
                        try:
                            llm_explainer = get_llm_explainer()
                            print(llm_explainer.explain(shap_result))
                        except Exception as e:
                            print(f"LLM error: {e}")
            else:
                print(f"Index out of range. Must be 0-{len(X_test)-1}")
        except ValueError:
            print("Invalid input. Enter a number or 'q'.")


if __name__ == "__main__":
    main()
