import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance as perm_importance
from sklearn.metrics import r2_score, mean_absolute_error
from src.regression_dataloader import RegressionDataLoader
from src.constants import *
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class DelayRandomForest:
    """Random Forest Regression"""

    def __init__(self, data_path=DATASET_PATH):
        """Initialize the dataloader"""
        self.dataloader = RegressionDataLoader(data_path)

    def tune_hyperparameters(self, X_train, y_train):
        """Grid search the best hyperparameters"""
        param_grid = {'n_estimators': [50, 100, 200],
                      'max_depth': [10, 20, None],
                      'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2]
                      }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

    def train_model(self):
        """Train Random Forest model"""
        X_train_encoded, X_test_encoded, y_train, y_test = self.dataloader.prepare_features(
            REGRESSION_FEATURE, REGRESSION_TARGET
        )
        best_params = self.tune_hyperparameters(X_train_encoded, y_train)
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train_encoded, y_train)

        y_pred_test = model.predict(X_test_encoded)
        test_r2 = r2_score(y_test, y_pred_test)
        features = list(REGRESSION_FEATURE)
        tree = model.feature_importances_
        perm = perm_importance(
            model, X_test_encoded, y_test,
            n_repeats=15, random_state=42, n_jobs=-1
        )
        sortindex = np.argsort(tree)[::-1]
        sorted_features = [features[i] for i in sortindex]
        results = {
            'test_r2': round(test_r2, 4),
            'mae': round(mean_absolute_error(y_test, y_pred_test), 2),
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'feature_importance': np.round(tree[sortindex], 4),
            'features': sorted_features,
            'perm_importance_mean': np.round(perm.importances_mean[sortindex], 4),
            'perm_importance_std': np.round(perm.importances_std[sortindex], 4),
        }
        return results
