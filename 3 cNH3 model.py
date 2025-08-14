#cNH3 model for the cooling and mature stage
import pandas as pd
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import joblib

# Import Data
dataset = pd.read_csv('MLSSNH3-cool.csv')

# Separate features (X), target (y), and groups (exp_id)
X = dataset.iloc[:, 0:8].values  # Assuming first 8 columns are features
y = dataset.iloc[:, 8].values    # Assuming 9th column is target
groups = dataset['exp_id'].values  # Assuming 'exp_id' is the column name for experimental IDs

# Initialize GroupShuffleSplit for train-test split
group_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
train_idx, test_idx = next(group_split.split(X, y, groups=groups))

# Split data
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scale features
scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

# SVR
svr_model = SVR()

# Define Bayesian optimization search space
search_spaces = {
    'C': Real(1e-1, 50.0, "log-uniform"),
    'epsilon': Real(0.1, 0.5, "uniform"),
    'kernel': Categorical(['rbf', 'linear']),
    'gamma': Real(0.1, 0.2, "log-uniform"),
}

# Initialize BayesSearchCV with GroupKFold for cross-validation
bayes_cv_tuner = BayesSearchCV(
    svr_model,
    search_spaces,
    n_iter=50,
    n_jobs=-1,
    cv=GroupKFold(n_splits=5),  # GroupKFold ensures groups are not split across folds
    scoring='neg_mean_squared_error',
    random_state=42
)

# Perform Bayesian optimization (pass groups to fit method)
bayes_cv_tuner.fit(rescaledX_train, y_train, groups=groups[train_idx])

# Output best parameters
print("Best parameters for SVR:", bayes_cv_tuner.best_params_)

# Evaluate on training and testing sets
best_svr = SVR(**bayes_cv_tuner.best_params_)
best_svr.fit(rescaledX_train, y_train)

# Training set evaluation
pre_train = best_svr.predict(rescaledX_train)
print('Training_data (SVR), MSE: %s' % mean_squared_error(y_train, pre_train))
print('Training_data (SVR), MAE: %s' % mean_absolute_error(y_train, pre_train))
print('Training_data (SVR), R2: %s' % r2_score(y_train, pre_train))

# Testing set evaluation
pre_test = best_svr.predict(rescaledX_test)
print('Testing_data (SVR), MSE: %s' % mean_squared_error(y_test, pre_test))
print('Testing_data (SVR), MAE: %s' % mean_absolute_error(y_test, pre_test))
print('Testing_data (SVR), R2: %s' % r2_score(y_test, pre_test))

# 保存模型
joblib.dump(best_svr, 'SVR_model_3.pkl')

# Data saving
# Training set - true value
train = pd.DataFrame(data=y_train)
train.to_csv('svr_train.csv')
# Training set - Predictive value
pretrain = pd.DataFrame(data=pre_train)
pretrain.to_csv('svr_predictions_train.csv')

# Testing set - true value
test = pd.DataFrame(data=y_test)
test.to_csv('svr_test.csv')
# Testing set - Predictive value
pretest = pd.DataFrame(data=pre_test)
pretest.to_csv('svr_predictions_test.csv')




# MLP
from sklearn.model_selection import cross_val_score, KFold
import optuna
import warnings
import numpy as np
np.random.seed(42)
warnings.filterwarnings('ignore')

cv = GroupKFold(n_splits=5)

def optimize_mlp(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50, 50), (100, 50), (100,100), (150, 100)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_float('alpha', 0.05, 0.4, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'max_iter': 2000,
        'random_state': 42
    }

    if params['solver'] == 'sgd':
        params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 0.01, 0.04, log=True)

    model = MLPRegressor(**params)

    score = -np.mean(cross_val_score(
        model,
        rescaledX_train,
        y_train,
        groups=groups[train_idx],
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1
    ))
    return score


print("\nOptimizing MLP...")
study_mlp = optuna.create_study(direction='minimize')
study_mlp.optimize(optimize_mlp, n_trials=50)


best_params_mlp = study_mlp.best_params
best_params_mlp.update({'random_state': 42, 'max_iter': 2000})
print("Best parameters for MLP:", best_params_mlp)


best_mlp = MLPRegressor(**best_params_mlp)
best_mlp.fit(rescaledX_train, y_train)


pre_train = best_mlp.predict(rescaledX_train)
print('Training_data (MLP), MSE: %.4f' % mean_squared_error(y_train, pre_train))
print('Training_data (MLP), MAE: %.4f' % mean_absolute_error(y_train, pre_train))
print('Training_data (MLP), R2: %.4f' % r2_score(y_train, pre_train))


pre_test = best_mlp.predict(rescaledX_test)
print('Testing_data (MLP), MSE: %.4f' % mean_squared_error(y_test, pre_test))
print('Testing_data (MLP), MAE: %.4f' % mean_absolute_error(y_test, pre_test))
print('Testing_data (MLP), R2: %.4f' % r2_score(y_test, pre_test))


joblib.dump(best_mlp, 'MLP_model_3.pkl')

# Data saving
# Training set - true value
train = pd.DataFrame(data=y_train)
train.to_csv('mlp_train.csv')
# Training set - Predictive value
pretrain = pd.DataFrame(data=pre_train)
pretrain.to_csv('mlp_predictions_train.csv')

# Testing set - true value
test = pd.DataFrame(data=y_test)
test.to_csv('mlp_test.csv')
# Testing set - Predictive value
pretest = pd.DataFrame(data=pre_test)
pretest.to_csv('mlp_predictions_test.csv')






# ExtraTrees
et_model = ExtraTreesRegressor(random_state=7)


et_search_spaces = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 9),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0, 'uniform')
}


et_bayes_cv = BayesSearchCV(
    et_model,
    et_search_spaces,
    n_iter=50,
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)


et_bayes_cv.fit(rescaledX_train, y_train, groups=groups[train_idx])


print("Best parameters for ExtraTrees:", et_bayes_cv.best_params_)
best_et = et_bayes_cv.best_estimator_


et_pre_train = best_et.predict(rescaledX_train)
print('Training_data (ET), MSE: %.4f' % mean_squared_error(y_train, et_pre_train))
print('Training_data (ET), MAE: %.4f' % mean_absolute_error(y_train, et_pre_train))
print('Training_data (ET), R2: %.4f' % r2_score(y_train, et_pre_train))


et_pre_test = best_et.predict(rescaledX_test)
print('Testing_data (ET), MSE: %.4f' % mean_squared_error(y_test, et_pre_test))
print('Testing_data (ET), MAE: %.4f' % mean_absolute_error(y_test, et_pre_test))
print('Testing_data (ET), R2: %.4f' % r2_score(y_test, et_pre_test))


joblib.dump(best_et, 'ET_model_3.pkl')

# Data saving
# Training set - true value
train = pd.DataFrame(data=y_train)
train.to_csv('et_train.csv')
# Training set - Predictive value
pretrain = pd.DataFrame(data=pre_train)
pretrain.to_csv('et_predictions_train.csv')

# Testing set - true value
test = pd.DataFrame(data=y_test)
test.to_csv('et_test.csv')
# Testing set - Predictive value
pretest = pd.DataFrame(data=pre_test)
pretest.to_csv('et_predictions_test.csv')



from catboost import CatBoostRegressor
# CatBoost
catboost_model = CatBoostRegressor(loss_function='MAE',verbose=0, random_state=42)


search_spaces = {
    'iterations': Integer(100, 1000),
    'depth': Integer(1, 5),
    'learning_rate': Real(0.01, 0.1, "log-uniform"),
    'l2_leaf_reg': Real(1.0, 10.0, "log-uniform"),
}


bayes_cv_tuner = BayesSearchCV(
    catboost_model,
    search_spaces,
    n_iter=50,
    n_jobs=-1,
    cv=GroupKFold(n_splits=5),
    scoring='neg_mean_squared_error',
    random_state=42
)


bayes_cv_tuner.fit(rescaledX_train, y_train, groups=groups[train_idx])


print("Best parameters for CatBoost:", bayes_cv_tuner.best_params_)


best_catboost = CatBoostRegressor(
    **bayes_cv_tuner.best_params_,
    loss_function='MAE',
    verbose=0,
    random_state=42
)
best_catboost.fit(rescaledX_train, y_train)

# Evaluate the training set
pre_train = best_catboost.predict(rescaledX_train)
print('Training_data (CatBoost), MSE: %s' % mean_squared_error(y_train, pre_train))
print('Training_data (CatBoost), MAE: %s' % mean_absolute_error(y_train, pre_train))
print('Training_data (CatBoost), R2: %s' % r2_score(y_train, pre_train))

# Evaluate the testing set
pre_test = best_catboost.predict(rescaledX_test)
print('Testing_data (CatBoost), MSE: %s' % mean_squared_error(y_test, pre_test))
print('Testing_data (CatBoost), MAE: %s' % mean_absolute_error(y_test, pre_test))
print('Testing_data (CatBoost), R2: %s' % r2_score(y_test, pre_test))


joblib.dump(best_catboost, 'CatBoost_model_3.pkl')

# Data saving
# Training set - true value
train = pd.DataFrame(data=y_train)
train.to_csv('catboost_train.csv')
# Training set - Predictive value
pretrain = pd.DataFrame(data=pre_train)
pretrain.to_csv('catboost_predictions_train.csv')

# Testing set - true value
test = pd.DataFrame(data=y_test)
test.to_csv('catboost_test.csv')
# Testing set - Predictive value
pretest = pd.DataFrame(data=pre_test)
pretest.to_csv('catboost_predictions_test.csv')



from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
# GPR model with fixed RBF kernel
gpr_model = GaussianProcessRegressor(
    kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
    n_restarts_optimizer=10
)

# Search space for GPR - only optimizing hyperparameters of RBF kernel
gpr_search_spaces = {
    'alpha': Real(1e-3, 0.07, 'log-uniform'),  # Added noise level
    'kernel__k1__constant_value': Real(1e-3, 1e3, 'log-uniform'),  # Constant kernel value
    'kernel__k2__length_scale': Real(1e-3, 1e3, 'log-uniform')  # RBF length scale
}

# Bayesian optimization with group-wise cross-validation
group_kfold = GroupKFold(n_splits=5)

gpr_bayes_cv = BayesSearchCV(
    gpr_model,
    gpr_search_spaces,
    n_iter=50,
    cv=group_kfold,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)

gpr_bayes_cv.fit(rescaledX_train, y_train, groups=groups[train_idx])

# Best parameters
print("Best parameters for GBR:", gpr_bayes_cv.best_params_)
best_gpr = gpr_bayes_cv.best_estimator_

# Training set evaluation
gpr_pre_train = best_gpr.predict(rescaledX_train)
print('Training_data (GPR), MSE:', mean_squared_error(y_train, gpr_pre_train))
print('Training_data (GPR), MAE:', mean_absolute_error(y_train, gpr_pre_train))
print('Training_data (GPR), R2:', r2_score(y_train, gpr_pre_train))

# Test set evaluation
gpr_pre_test = best_gpr.predict(rescaledX_test)
print('Testing_data (GPR), MSE:', mean_squared_error(y_test, gpr_pre_test))
print('Testing_data (GPR), MAE:', mean_absolute_error(y_test, gpr_pre_test))
print('Testing_data (GPR), R2:', r2_score(y_test, gpr_pre_test))


joblib.dump(best_gpr, 'GPR_model_3.pkl')


import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# Import Data
dataset = pd.read_csv('MLSSNH3-cool.csv')


X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values
groups = dataset['exp_id'].values

# Initialize GroupShuffleSplit for train-test split
group_split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
train_idx, test_idx = next(group_split.split(X, y, groups=groups))


X_train_raw, X_test = X[train_idx], X[test_idx]
y_train_raw, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]



def remove_outliers_isolation_forest(X, y, groups, feature_columns, contamination=0.05, random_state=42):
    train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    train_df['target'] = y
    train_df['exp_id'] = groups

    # 初始化Isolation Forest
    iso_forest = IsolationForest(contamination=contamination,
                                 random_state=random_state)


    iso_forest.fit(X)


    outliers_mask = iso_forest.predict(X) == -1


    clean_X = X[~outliers_mask]
    clean_y = y[~outliers_mask]
    clean_groups = groups[~outliers_mask]

    print(f"remove {outliers_mask.sum()} rows ({(outliers_mask.sum() / len(X)) * 100:.2f}%) outliers")
    return clean_X, clean_y, clean_groups, ~outliers_mask



feature_columns = list(range(8))
X_train, y_train, groups_train, train_mask = remove_outliers_isolation_forest(
    X_train_raw, y_train_raw, groups_train, feature_columns
)


scaler = RobustScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
rescaledX_test = scaler.transform(X_test)

cv = GroupKFold(n_splits=5)

def optimize_mlp(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50, 50), (100, 50), (100,100), (150, 100)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_float('alpha', 0.05, 0.4, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'max_iter': 2000,
        'random_state': 42
    }

    if params['solver'] == 'sgd':
        params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 0.01, 0.04, log=True)

    model = MLPRegressor(**params)

    score = -np.mean(cross_val_score(
        model,
        rescaledX_train,
        y_train,
        groups=groups_train,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1
    ))
    return score


print("\nOptimizing MLP...")
study_mlp = optuna.create_study(direction='minimize')
study_mlp.optimize(optimize_mlp, n_trials=50)


best_params_mlp = study_mlp.best_params
best_params_mlp.update({'random_state': 42, 'max_iter': 2000})
print("Best parameters for MLP:", best_params_mlp)


best_mlp = MLPRegressor(**best_params_mlp)
best_mlp.fit(rescaledX_train, y_train)


pre_train = best_mlp.predict(rescaledX_train)
print('Training_data (MLP), MSE: %.4f' % mean_squared_error(y_train, pre_train))
print('Training_data (MLP), MAE: %.4f' % mean_absolute_error(y_train, pre_train))
print('Training_data (MLP), R2: %.4f' % r2_score(y_train, pre_train))


pre_test = best_mlp.predict(rescaledX_test)
print('Testing_data (MLP), MSE: %.4f' % mean_squared_error(y_test, pre_test))
print('Testing_data (MLP), MAE: %.4f' % mean_absolute_error(y_test, pre_test))
print('Testing_data (MLP), R2: %.4f' % r2_score(y_test, pre_test))