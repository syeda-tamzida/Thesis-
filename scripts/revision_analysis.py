"""Revision analysis for Reviewer 2 (R2.1, R2.2), Reviewer 3 (R3.3, R3.4), Reviewer 4 (percentage error).

Same preprocessing and model settings as data_process_ML_results.ipynb:
- drop rows with missing values (250 -> 237)
- 80/20 split, StandardScaler fitted on the training part only
- KNN: GridSearchCV(k 1-20, weights, p), 5-fold, r2
- RF : GridSearchCV(n_estimators 30/50/70, max_depth 3-5, min_samples_split 2/5, min_samples_leaf 1/2)
- SVR: rbf, C=100, gamma='scale', epsilon=0.01
- Linear Regression, Bayesian Ridge (defaults)
Repeated over 30 random splits (seeds 0-29) for every feature set.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data_path, out_dir = sys.argv[1], sys.argv[2]
df = pd.read_csv(data_path).dropna().reset_index(drop=True)
y = df['half-cell_potential_v'].values
T, H, RES, RO = 'temperature_c', 'humidity_%', 'surface_resistivity', 'surface_roughness'
FEATURE_SETS = {
    'Resistivity only': [RES],
    'Resistivity + Roughness': [RES, RO],
    'Resistivity + Temp + Humidity': [RES, T, H],
    'All four inputs': [T, H, RES, RO],
}

def make_models():
    return {
        'KNN': GridSearchCV(Pipeline([('s', StandardScaler()), ('m', KNeighborsRegressor())]),
                            {'m__n_neighbors': list(range(1, 21)), 'm__weights': ['uniform', 'distance'], 'm__p': [1, 2]},
                            scoring='r2', cv=5, n_jobs=-1),
        'Random Forest': GridSearchCV(Pipeline([('s', StandardScaler()), ('m', RandomForestRegressor(random_state=42))]),
                            {'m__n_estimators': [30, 50, 70], 'm__max_depth': [3, 4, 5],
                             'm__min_samples_split': [2, 5], 'm__min_samples_leaf': [1, 2]},
                            scoring='r2', cv=5, n_jobs=-1),
        'SVR': Pipeline([('s', StandardScaler()), ('m', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01))]),
        'Bayesian Ridge': Pipeline([('s', StandardScaler()), ('m', BayesianRidge())]),
        'Linear Regression': Pipeline([('s', StandardScaler()), ('m', LinearRegression())]),
    }

def metrics(yt, yp):
    mse = mean_squared_error(yt, yp)
    return dict(R2=r2_score(yt, yp), MAE=mean_absolute_error(yt, yp), MSE=mse, RMSE=np.sqrt(mse),
                MAPE=np.mean(np.abs((yt - yp) / yt)) * 100,
                NRMSE=np.sqrt(mse) / (y.max() - y.min()) * 100)  # % of full HCP range

rows, best_params = [], []
for fs_name, cols in FEATURE_SETS.items():
    X = df[cols].values
    for seed in range(30):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        for mname, model in make_models().items():
            model.fit(Xtr, ytr)
            m = metrics(yte, model.predict(Xte))
            rows.append(dict(feature_set=fs_name, model=mname, seed=seed, **m))
            if fs_name == 'All four inputs' and hasattr(model, 'best_params_'):
                best_params.append(dict(model=mname, seed=seed, **{k.replace('m__', ''): v for k, v in model.best_params_.items()}))
    print('done', fs_name, flush=True)

res = pd.DataFrame(rows)
res.to_csv(f'{out_dir}/repeated_splits_raw.csv', index=False)
summ = res.groupby(['feature_set', 'model'])[['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'NRMSE']].agg(['mean', 'std'])
summ.to_csv(f'{out_dir}/repeated_splits_summary.csv')
pd.DataFrame(best_params).to_csv(f'{out_dir}/best_params_all_inputs.csv', index=False)

# Reproduce original Table 6 split (seed 42, all inputs) and add percentage error
X = df[FEATURE_SETS['All four inputs']].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
t6 = []
for mname, model in make_models().items():
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    t6.append(dict(model=mname, **metrics(yte, p),
                   params=json.dumps(getattr(model, 'best_params_', {}))))
    if mname == 'KNN':
        pd.DataFrame({'actual': yte, 'pred': p}).to_csv(f'{out_dir}/knn_seed42_preds.csv', index=False)
pd.DataFrame(t6).to_csv(f'{out_dir}/table6_seed42.csv', index=False)

# Leave-one-measurement-session-out (sessions = unique temperature/humidity pairs)
groups = df[T].astype(str) + '_' + df[H].astype(str)
gk = GroupKFold(n_splits=groups.nunique())
g_rows = []
for fs_name, cols in FEATURE_SETS.items():
    X = df[cols].values
    for mname in ['KNN', 'Random Forest', 'Linear Regression']:
        preds = np.zeros_like(y)
        for tr, te in gk.split(X, y, groups):
            model = make_models()[mname]
            model.fit(X[tr], y[tr]); preds[te] = model.predict(X[te])
        g_rows.append(dict(feature_set=fs_name, model=mname, **metrics(y, preds)))
pd.DataFrame(g_rows).to_csv(f'{out_dir}/leave_session_out.csv', index=False)
print('finished')
