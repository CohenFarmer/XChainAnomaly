import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, average_precision_score
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import optuna

from machineLearning.supervisedModels import test_model


def load_data(path: str,
			  drop_cols: list[str] | None = None,
			  label_col: str = 'label',
			  test_size: float = 0.33,
			  seed: int = 40):
	df = pd.read_csv(path)
	X = df.drop(columns=[label_col] + (drop_cols or []))
	y = df[label_col]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=seed, stratify=y
	)
	return X_train, X_test, y_train, y_test


def compute_sample_weights(y):
	classes = np.unique(y)
	weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
	w_map = {c: w for c, w in zip(classes, weights)}
	return np.array([w_map[label] for label in y])


def tune_xgboost(X_train, y_train, X_test, y_test, seed=40):
	model = xgb.XGBClassifier(objective='binary:logistic', random_state=seed)

	param_dist = {
		'max_depth': [4, 6, 8, 10, 12],
		'learning_rate': [0.03, 0.05, 0.075, 0.1],
		'n_estimators': [300, 600, 900, 1200],
		'min_child_weight': [1, 3, 5, 7],
		'subsample': [0.7, 0.85, 1.0],
		'colsample_bytree': [0.6, 0.8, 1.0],
		'reg_lambda': [0.5, 1.0, 2.0],
		'reg_alpha': [0.0, 0.1, 0.5],
		'gamma': [0, 1, 5],
	}

	cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
	scorer = 'average_precision'
	sw = compute_sample_weights(y_train)

	search = RandomizedSearchCV(
		estimator=model,
		param_distributions=param_dist,
		n_iter=25,
		scoring=scorer,
		cv=cv,
		verbose=1,
		n_jobs=-1,
		random_state=seed,
	)
	search.fit(X_train, y_train, sample_weight=sw)
	print('Best XGBoost params:', search.best_params_)
	best = search.best_estimator_
	test_model(best, X_test, y_test, binary=True)
	return best


def tune_xgboost_bayes(X_train, y_train, X_test, y_test, seed=40, n_trials=60):
	"""Bayesian hyperparameter optimization for XGBoost using Optuna.

	Optimizes average_precision (PR-AUC) via 5-fold Stratified CV
	with class-balanced sample weights per fold.
	"""
	if optuna is None:
		raise RuntimeError("Optuna is not installed. Run: pip install optuna")

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

	def objective(trial: 'optuna.trial.Trial'):
		params = {
			'objective': 'binary:logistic',
			'max_depth': trial.suggest_int('max_depth', 4, 12),
			'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15, log=True),
			'n_estimators': trial.suggest_int('n_estimators', 600, 2000),
			'min_child_weight': trial.suggest_int('min_child_weight', 1, 9),
			'subsample': trial.suggest_float('subsample', 0.8, 1.0),
			'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
			'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.5, log=True),
			'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.2),
			'gamma': trial.suggest_float('gamma', 0.0, 0.5),
			'random_state': seed,
			'n_jobs': -1,
		}

		ap_scores = []
		for tr_idx, val_idx in cv.split(X_train, y_train):
			X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
			y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

			# Balanced sample weights per fold
			classes = np.unique(y_tr)
			weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
			w_map = {c: w for c, w in zip(classes, weights)}
			sw_fold = np.array([w_map[label] for label in y_tr])

			clf = xgb.XGBClassifier(**params)
			clf.fit(
				X_tr,
				y_tr,
				sample_weight=sw_fold,
				eval_set=[(X_val, y_val)],
				verbose=False,
			)
			y_proba = clf.predict_proba(X_val)[:, 1]
			ap = average_precision_score(y_val, y_proba)
			ap_scores.append(ap)

		return float(np.mean(ap_scores))

	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
	print('Best XGBoost (Bayes) params:', study.best_params)

	# Train final model on full training data with best params
	best_params = {
		**study.best_params,
		'objective': 'binary:logistic',
		'random_state': seed,
		'n_jobs': -1,
	}
	classes = np.unique(y_train)
	weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
	w_map = {c: w for c, w in zip(classes, weights)}
	sw = np.array([w_map[label] for label in y_train])

	best_model = xgb.XGBClassifier(**best_params)
	best_model.fit(X_train, y_train, sample_weight=sw, verbose=False)
	test_model(best_model, X_test, y_test, binary=True)
	return best_model


def tune_random_forest(X_train, y_train, X_test, y_test, seed=40):
	model = RandomForestClassifier(random_state=seed, n_jobs=-1)

	param_dist = {
		'n_estimators': [300, 600, 900],
		'max_depth': [None, 10, 20, 40],
		'max_features': ['sqrt', 'log2', None],
		'min_samples_split': [2, 5, 10],
		'min_samples_leaf': [1, 2, 4],
		'bootstrap': [True],
	}

	cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
	scorer = 'average_precision'
	sw = compute_sample_weights(y_train)

	search = RandomizedSearchCV(
		estimator=model,
		param_distributions=param_dist,
		n_iter=20,
		scoring=scorer,
		cv=cv,
		verbose=1,
		n_jobs=-1,
		random_state=seed,
	)
	search.fit(X_train, y_train, sample_weight=sw)
	print('Best RandomForest params:', search.best_params_)
	best = search.best_estimator_
	test_model(best, X_test, y_test, binary=True)
	return best


def tune_logistic_regression(X_train, y_train, X_test, y_test, seed=40):
	pipe = Pipeline([
		('scaler', StandardScaler()),
		('logreg', LogisticRegression(random_state=seed, solver='lbfgs', max_iter=1000, multi_class='auto')),
	])

	param_dist = {
		'logreg__C': [0.1, 0.5, 1.0, 2.0, 5.0],
		'logreg__penalty': ['l2'],
		'logreg__solver': ['lbfgs'],
	}

	cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
	scorer = 'average_precision'
	sw = compute_sample_weights(y_train)

	search = RandomizedSearchCV(
		estimator=pipe,
		param_distributions=param_dist,
		n_iter=15,
		scoring=scorer,
		cv=cv,
		verbose=1,
		n_jobs=-1,
		random_state=seed,
	)
	search.fit(X_train, y_train, logreg__sample_weight=sw)
	print('Best LogisticRegression params:', search.best_params_)
	best = search.best_estimator_
	test_model(best, X_test, y_test, binary=True)
	return best


def tune_random_forest_bayes(X_train, y_train, X_test, y_test, seed=40, n_trials=50):
	"""Bayesian hyperparameter optimization for RandomForest using Optuna.

	Optimizes average_precision (PR-AUC) via 5-fold Stratified CV
	with class-balanced sample weights per fold.
	"""
	if optuna is None:
		raise RuntimeError("Optuna is not installed. Run: pip install optuna")

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

	def objective(trial: 'optuna.trial.Trial'):
		max_depth_choice = trial.suggest_categorical('max_depth', [None, 10, 20, 40])
		params = {
			'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
			'max_depth': max_depth_choice,
			'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
			'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
			'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
			'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
			'random_state': seed,
			'n_jobs': -1,
		}

		ap_scores = []
		for tr_idx, val_idx in cv.split(X_train, y_train):
			X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
			y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

			classes = np.unique(y_tr)
			weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
			w_map = {c: w for c, w in zip(classes, weights)}
			sw_fold = np.array([w_map[label] for label in y_tr])

			clf = RandomForestClassifier(**params)
			clf.fit(X_tr, y_tr, sample_weight=sw_fold)
			y_proba = clf.predict_proba(X_val)[:, 1]
			ap = average_precision_score(y_val, y_proba)
			ap_scores.append(ap)

		return float(np.mean(ap_scores))

	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
	print('Best RandomForest (Bayes) params:', study.best_params)

	best_params = {**study.best_params, 'random_state': seed, 'n_jobs': -1}
	classes = np.unique(y_train)
	weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
	w_map = {c: w for c, w in zip(classes, weights)}
	sw = np.array([w_map[label] for label in y_train])

	best_model = RandomForestClassifier(**best_params)
	best_model.fit(X_train, y_train, sample_weight=sw)
	test_model(best_model, X_test, y_test, binary=True)
	return best_model


def tune_logistic_regression_bayes(X_train, y_train, X_test, y_test, seed=40, n_trials=40):
	"""Bayesian hyperparameter optimization for LogisticRegression using Optuna.

	Optimizes average_precision via 5-fold Stratified CV with a scaler+LR pipeline.
	"""
	if optuna is None:
		raise RuntimeError("Optuna is not installed. Run: pip install optuna")

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

	def objective(trial: 'optuna.trial.Trial'):
		C_val = trial.suggest_float('logreg__C', 1e-2, 10.0, log=True)
		solver = trial.suggest_categorical('logreg__solver', ['lbfgs'])
		pipe = Pipeline([
			('scaler', StandardScaler()),
			('logreg', LogisticRegression(random_state=seed, solver=solver, max_iter=1000, multi_class='auto', C=C_val, penalty='l2')),
		])

		ap_scores = []
		for tr_idx, val_idx in cv.split(X_train, y_train):
			X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
			y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

			classes = np.unique(y_tr)
			weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
			w_map = {c: w for c, w in zip(classes, weights)}
			sw_fold = np.array([w_map[label] for label in y_tr])

			pipe.fit(X_tr, y_tr, logreg__sample_weight=sw_fold)
			y_proba = pipe.predict_proba(X_val)[:, 1]
			ap = average_precision_score(y_val, y_proba)
			ap_scores.append(ap)

		return float(np.mean(ap_scores))

	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
	print('Best LogisticRegression (Bayes) params:', study.best_params)

	C_val = study.best_params.get('logreg__C', 1.0)
	solver = study.best_params.get('logreg__solver', 'lbfgs')
	pipe = Pipeline([
		('scaler', StandardScaler()),
		('logreg', LogisticRegression(random_state=seed, solver=solver, max_iter=1000, multi_class='auto', C=C_val, penalty='l2')),
	])

	classes = np.unique(y_train)
	weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
	w_map = {c: w for c, w in zip(classes, weights)}
	sw = np.array([w_map[label] for label in y_train])

	pipe.fit(X_train, y_train, logreg__sample_weight=sw)
	test_model(pipe, X_test, y_test, binary=True)
	return pipe


def main():
	data_path = 'features/datasets/cross_chain_labeled_transactions_enriched_probs.csv'
	seed = 40
	test_size = 0.33
	drop_cols = ['source_index', 'src_from_address', 'recipient', 'src_blockchain', 'dst_blockchain']

	X_train, X_test, y_train, y_test = load_data(
		data_path, drop_cols=drop_cols, test_size=test_size, seed=seed
	)

	print('Tuning XGBoost...')
	# Prefer Bayesian optimization (Optuna) for better sample efficiency
	if optuna is not None:
		tune_xgboost_bayes(X_train, y_train, X_test, y_test, seed=seed, n_trials=1)
	else:
		tune_xgboost(X_train, y_train, X_test, y_test, seed=seed)

	#print('\nTuning RandomForest...')
	if optuna is not None:
		tune_random_forest_bayes(X_train, y_train, X_test, y_test, seed=seed, n_trials=10)
	else:
		tune_random_forest(X_train, y_train, X_test, y_test, seed=seed)

	#print('\nTuning LogisticRegression...')
	if optuna is not None:
		tune_logistic_regression_bayes(X_train, y_train, X_test, y_test, seed=seed, n_trials=10)
	else:
		tune_logistic_regression(X_train, y_train, X_test, y_test, seed=seed)


if __name__ == '__main__':
	main()

