
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class RegressionDataLoader:
    """Prepare data for various regression models"""

    def __init__(self, data_path):
        """Initialize the class-wise parameters"""
        self.data_path = data_path

    def load_and_preprocess_data(self):
        """Load data from csv"""
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna()
        return self.df
    
    def prepare_features(self, features, target):
        """
        Encode categorical features
        """
        self.load_and_preprocess_data()
        encoders = {}
        X = self.df[features].copy()
        y = self.df[target].copy()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        if 'carrier' not in encoders:
            encoders['carrier'] = LabelEncoder()
            X_train_encoded['carrier'] = encoders['carrier'].fit_transform(X_train['carrier'])
        else:
            X_train_encoded['carrier'] = encoders['carrier'].transform(X_train['carrier'])
        X_test_encoded['carrier'] = encoders['carrier'].transform(X_test['carrier'])
        
        if 'airport' not in encoders:
            encoders['airport'] = LabelEncoder()
            X_train_encoded['airport'] = encoders['airport'].fit_transform(X_train['airport'])
        else:
            X_train_encoded['airport'] = encoders['airport'].transform(X_train['airport'])
        X_test_encoded['airport'] = encoders['airport'].transform(X_test['airport'])
        
        return X_train_encoded, X_test_encoded, y_train, y_test
    