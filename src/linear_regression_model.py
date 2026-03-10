import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.regression_dataloader import RegressionDataLoader
from src.constants import *
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class DelayLinearRegression:
    """Linear Regression"""
    
    def __init__(self, data_path = DATASET_PATH):
        """Initialize the dataloader with data path"""
        self.dataloader = RegressionDataLoader(data_path)
    
    
    def train_model(self):
        """Train Linear Regression model for total delay prediction"""
        X_train_encoded, X_test_encoded, y_train, y_test = self.dataloader.prepare_features(REGRESSION_FEATURE, REGRESSION_TARGET)
        model = LinearRegression()
        model.fit(X_train_encoded, y_train)
        y_pred_train = model.predict(X_train_encoded)
        y_pred_test = model.predict(X_test_encoded)
        test_r2 = r2_score(y_test, y_pred_test)
        results = {
            'test_r2': round(test_r2, 1),
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'coefficients': np.round(model.coef_, 1),
            'features': REGRESSION_FEATURE
        }
        return results