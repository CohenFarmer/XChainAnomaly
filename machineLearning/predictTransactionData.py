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

dataset = pd.read_csv('features/datasets/cross_chain_labeled_transactions_enriched_probs.csv')

test_size = 0.33
seed = 40

X = dataset.drop(columns=['label', 'source_index', 'src_from_address', 'recipient', 'src_blockchain', 'dst_blockchain'])

y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

classes = np.unique(y_train)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_to_weight = {c: w for c, w in zip(classes, class_weights)}
sample_weights = np.array([class_to_weight[label] for label in y_train])

#xgbModel = train_xgboost_model(X_train, y_train, sample_weights=sample_weights, binary=True)
#lrModel = train_logistic_regression_model(X_train, y_train, sample_weights=sample_weights, binary=True)
rfModel = train_random_forest_model(X_train, y_train, sample_weights=sample_weights, binary=True)
#test_model(xgbModel, X_test, y_test, binary=True)
#test_model(lrModel, X_test, y_test, binary=True)
test_model(rfModel, X_test, y_test, binary=True, threshold=0.45)
test_model(rfModel, X_test, y_test, binary=True, threshold=0.4)
test_model(rfModel, X_test, y_test, binary=True, threshold=0.35)
test_model(rfModel, X_test, y_test, binary=True, threshold=0.3)
test_model(rfModel, X_test, y_test, binary=True, threshold=0.25)