

import shap
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import json


CLASS_NAMES = ['Non-Malicious', 'Phishing', 'Exploit', 'Sanctioned', 'Tornado']


class TransactionExplainer:

    
    def __init__(self, multi_output_model, feature_names: List[str]):

        self.model = multi_output_model
        self.feature_names = feature_names
        self.src_explainer = shap.TreeExplainer(multi_output_model.estimators_[0])
        self.rec_explainer = shap.TreeExplainer(multi_output_model.estimators_[1])
    
    def get_shap_explanation(self, X_instance) -> Dict[str, Any]:

        X = X_instance.values.reshape(1, -1) if hasattr(X_instance, 'values') else np.array(X_instance).reshape(1, -1)
        

        pred = self.model.predict(X)[0]
        src_class, rec_class = int(pred[0]), int(pred[1])
        

        src_proba = None
        rec_proba = None
        if hasattr(self.model.estimators_[0], 'predict_proba'):
            src_proba = self.model.estimators_[0].predict_proba(X)[0].tolist()
            rec_proba = self.model.estimators_[1].predict_proba(X)[0].tolist()
        
        src_shap = self.src_explainer.shap_values(X)
        rec_shap = self.rec_explainer.shap_values(X)
        
        n_features = len(self.feature_names)
        
        if isinstance(src_shap, list):
            src_contributions = np.array(src_shap[src_class]).flatten()[:n_features]
            rec_contributions = np.array(rec_shap[rec_class]).flatten()[:n_features]
        elif isinstance(src_shap, np.ndarray) and src_shap.ndim == 3:
            src_contributions = src_shap[0, :, src_class].flatten()[:n_features]
            rec_contributions = rec_shap[0, :, rec_class].flatten()[:n_features]
        else:
            src_contributions = np.array(src_shap).flatten()[:n_features]
            rec_contributions = np.array(rec_shap).flatten()[:n_features]
        
        feature_values = np.array(X).flatten()[:n_features]
        

        src_features = pd.DataFrame({
            'feature': self.feature_names,
            'value': feature_values,
            'contribution': src_contributions
        }).sort_values('contribution', key=abs, ascending=False)
        
        rec_features = pd.DataFrame({
            'feature': self.feature_names,
            'value': feature_values,
            'contribution': rec_contributions
        }).sort_values('contribution', key=abs, ascending=False)
        
        return {
            'source': {
                'predicted_class': src_class,
                'predicted_label': CLASS_NAMES[src_class],
                'probabilities': dict(zip(CLASS_NAMES, src_proba)) if src_proba else None,
                'feature_contributions': src_features.to_dict('records'),
                'base_value': float(self.src_explainer.expected_value[src_class]) if isinstance(self.src_explainer.expected_value, np.ndarray) else float(self.src_explainer.expected_value)
            },
            'recipient': {
                'predicted_class': rec_class,
                'predicted_label': CLASS_NAMES[rec_class],
                'probabilities': dict(zip(CLASS_NAMES, rec_proba)) if rec_proba else None,
                'feature_contributions': rec_features.to_dict('records'),
                'base_value': float(self.rec_explainer.expected_value[rec_class]) if isinstance(self.rec_explainer.expected_value, np.ndarray) else float(self.rec_explainer.expected_value)
            },
            'feature_values': dict(zip(self.feature_names, feature_values.tolist()))
        }


class LLMExplainer:

    
    SYSTEM_PROMPT = """You are an expert blockchain security analyst explaining cross-chain transaction anomaly detection results.

You will receive:
1. Transaction features (fees, amounts, blockchain info)
2. Model predictions for source and recipient address classifications
3. SHAP feature contributions showing which features influenced each prediction

Your job is to explain in clear, professional language:
- What the model believes about who is on each side of the transaction
- WHY the model believes this, based on the key contributing features
- Any notable patterns or red flags

Keep explanations concise but informative. Use specific numbers from the features.
Do not speculate beyond what the data shows. If confidence is low, say so.

Class meanings:
- Non-Malicious: Regular user or legitimate service
- Phishing: Address associated with phishing scams
- Exploit: Address linked to protocol exploits or hacks  
- Sanctioned: OFAC-sanctioned address
- Tornado: Tornado Cash mixer user/depositor"""

    def __init__(self, llm_client, model_name: str = "gpt-4"):
    
        self.client = llm_client
        self.model_name = model_name
    
    def _format_features_for_prompt(self, explanation: Dict[str, Any], top_n: int = 5) -> str:
    
        features = explanation['feature_values']
        src = explanation['source']
        rec = explanation['recipient']
       
        feature_str = "TRANSACTION FEATURES:\n"
        for name, value in features.items():
            feature_str += f"  {name}: {value:.4f}\n"

        src_str = f"\nSOURCE ADDRESS PREDICTION:\n"
        src_str += f"  Predicted class: {src['predicted_label']}\n"
        if src['probabilities']:
            src_str += f"  Confidence: {src['probabilities'][src['predicted_label']]*100:.1f}%\n"
            src_str += f"  All probabilities: {json.dumps({k: f'{v*100:.1f}%' for k,v in src['probabilities'].items()})}\n"
        src_str += f"  Top contributing features:\n"
        for feat in src['feature_contributions'][:top_n]:
            direction = "+" if feat['contribution'] > 0 else ""
            src_str += f"    {feat['feature']}: value={feat['value']:.4f}, contribution={direction}{feat['contribution']:.4f}\n"

        rec_str = f"\nRECIPIENT ADDRESS PREDICTION:\n"
        rec_str += f"  Predicted class: {rec['predicted_label']}\n"
        if rec['probabilities']:
            rec_str += f"  Confidence: {rec['probabilities'][rec['predicted_label']]*100:.1f}%\n"
            rec_str += f"  All probabilities: {json.dumps({k: f'{v*100:.1f}%' for k,v in rec['probabilities'].items()})}\n"
        rec_str += f"  Top contributing features:\n"
        for feat in rec['feature_contributions'][:top_n]:
            direction = "+" if feat['contribution'] > 0 else ""
            rec_str += f"    {feat['feature']}: value={feat['value']:.4f}, contribution={direction}{feat['contribution']:.4f}\n"
        
        return feature_str + src_str + rec_str
    
    def explain(self, shap_explanation: Dict[str, Any], top_features: int = 5) -> str:
        
        user_prompt = f"""Analyze this cross-chain transaction and explain the model's predictions:

{self._format_features_for_prompt(shap_explanation, top_features)}

Provide a clear explanation of:
1. What the model believes about the source address and why
2. What the model believes about the recipient address and why
3. Overall assessment of this transaction's risk profile"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    
    def explain_batch(self, explanations: List[Dict[str, Any]], top_features: int = 5) -> List[str]:
        
        return [self.explain(exp, top_features) for exp in explanations]


class LocalLLMExplainer(LLMExplainer):

    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3"):

        import requests
        self.base_url = base_url
        self.model_name = model_name
        self._requests = requests
    
    def explain(self, shap_explanation: Dict[str, Any], top_features: int = 5) -> str:
    
        
        user_prompt = f"""Analyze this cross-chain transaction and explain the model's predictions:

{self._format_features_for_prompt(shap_explanation, top_features)}

Provide a clear explanation of:
1. What the model believes about the source address and why
2. What the model believes about the recipient address and why  
3. Overall assessment of this transaction's risk profile"""

        full_prompt = f"{self.SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
        
        response = self._requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 800
                }
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")


def create_explainer_pipeline(multi_output_model, feature_names: List[str], 
                               llm_provider: str = "openai",
                               **llm_kwargs) -> tuple:
   
    tx_explainer = TransactionExplainer(multi_output_model, feature_names)
    
    if llm_provider == "openai":
        from openai import OpenAI
        client = OpenAI(**llm_kwargs)
        llm_explainer = LLMExplainer(client, model_name=llm_kwargs.get('model', 'gpt-4'))
    
    elif llm_provider == "anthropic":
        # Anthropic has slightly different API structure
        from anthropic import Anthropic
        client = Anthropic(**llm_kwargs)
        llm_explainer = AnthropicExplainer(client, model_name=llm_kwargs.get('model', 'claude-3-sonnet-20240229'))
    
    elif llm_provider == "local":
        llm_explainer = LocalLLMExplainer(
            base_url=llm_kwargs.get('base_url', 'http://localhost:11434'),
            model_name=llm_kwargs.get('model', 'llama3')
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")
    
    return tx_explainer, llm_explainer


class AnthropicExplainer(LLMExplainer):
    
    def __init__(self, client, model_name: str = "claude-3-sonnet-20240229"):
        self.client = client
        self.model_name = model_name
    
    def explain(self, shap_explanation: Dict[str, Any], top_features: int = 5) -> str:
        user_prompt = f"""Analyze this cross-chain transaction and explain the model's predictions:

{self._format_features_for_prompt(shap_explanation, top_features)}

Provide a clear explanation of:
1. What the model believes about the source address and why
2. What the model believes about the recipient address and why
3. Overall assessment of this transaction's risk profile"""

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=800,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.content[0].text


# Convenience function for quick explanations
def explain_transaction(model, X_instance, feature_names: List[str], 
                        llm_client=None, llm_provider: str = "openai") -> Dict[str, Any]:

    tx_explainer = TransactionExplainer(model, feature_names)
    shap_result = tx_explainer.get_shap_explanation(X_instance)
    
    result = {
        'shap_explanation': shap_result,
        'natural_language': None
    }
    
    if llm_client:
        if llm_provider == "openai":
            llm_explainer = LLMExplainer(llm_client)
        elif llm_provider == "anthropic":
            llm_explainer = AnthropicExplainer(llm_client)
        elif llm_provider == "local":
            llm_explainer = LocalLLMExplainer()
        
        result['natural_language'] = llm_explainer.explain(shap_result)
    
    return result


if __name__ == "__main__":
    print("Explainability module loaded.")
    print("Usage example:")
    print("""
    from machineLearning.explainability import TransactionExplainer, LLMExplainer
    from openai import OpenAI
    import joblib
    
    # Load your trained Model 3
    model = joblib.load('machineLearning/models/multi_output_model.joblib')
    
    # Create explainer
    explainer = TransactionExplainer(model, feature_names=X.columns.tolist())
    
    # Get SHAP explanation for a transaction
    shap_result = explainer.get_shap_explanation(X_test.iloc[0])
    
    # Use LLM for natural language explanation
    client = OpenAI()
    llm_explainer = LLMExplainer(client)
    explanation = llm_explainer.explain(shap_result)
    print(explanation)
    """)
