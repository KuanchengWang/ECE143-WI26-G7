import os
import pandas as pd
import kagglehub
from pathlib import Path
from typing import Optional

from src import constants

class Preprocessing:
    """Handles data downloading, cleaning, and saving for the flight delay analysis.
    Attributes:
        df (pd.DataFrame): The raw dataset from the CSV file.
        clean_df (pd.DataFrame): The cleaned dataset after handling missing values.
        delay_cols (list): List of columns related to delay causes.
    """

    def __init__(self):
        """Initializes the Preprocessing class with default values."""
        self.df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.delay_cols = ['arr_delay', 
                           'carrier_delay', 
                           'weather_delay', 
                           'nas_delay', 
                           'security_delay', 
                           'late_aircraft_delay']
        
    def make_dir(self):
        """Creates necessary folders structure."""
        for _d in [constants.DATASET_DIR, constants.SAVE_DIR]:
            assert isinstance(_d, str), f"Directory path must be a string, got {type(_d)}"
            os.makedirs(_d, exist_ok=True)
        
    def download_data(self, dataset_handle = constants.DATASET_HANDLE):
        """Downloads the dataset from Kaggle and loads it into a DataFrame.
        Args:
            dataset_handle (str): The Kaggle dataset handle to download from.
        Returns:
            pd.DataFrame: The raw dataset loaded into a DataFrame.
        """
        assert isinstance(dataset_handle, str), "dataset_handle must be a string"
        try:
            path = kagglehub.dataset_download(dataset_handle)
            dataset_dir = Path(path)
            
            csv_files = list(dataset_dir.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError("No CSV file found in the downloaded dataset folder.")
            
            file_path = csv_files[0]
            self.df = pd.read_csv(file_path)
            assert self.df is not None and not self.df.empty, "Failed to load data: DataFrame is None or empty"
            
        except Exception as e:
            raise Exception(f"Failed to download or load data: {e}")
        return self.df

    def get_clean_data(self, df = None):
        """Cleans the dataset by handling missing values.
        Args:
            df (pd.DataFrame): The raw dataset to be cleaned.
        Returns:
            pd.DataFrame: The cleaned dataset with missing values handled.
        """
        if df is None:
            df = self.download_data()
        
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert all(col in df.columns for col in self.delay_cols), f"DataFrame must contain delay columns: {self.delay_cols}"
        
        self.df_clean = df.copy()

        missing = self.df_clean.isnull().sum()
        missing_pct = (missing / len(self.df_clean) * 100).round(2)
        missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        self.df_clean = self.df_clean.dropna(subset = self.delay_cols).copy()
        assert self.df_clean is not None and not self.df_clean.empty, "Cleaning resulted in empty DataFrame"
        return self.df_clean

    def save_data(self, df, filename: str = constants.DATASET_PATH):
        """Saves the cleaned dataset to a CSV file.
        Args:
            df (pd.DataFrame): The cleaned dataset to be saved.
            filename (str): The path where the cleaned dataset will be saved.
        """
        assert df is not None, "Dataframe is empty. Run download_data() and get_clean_data() first."
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(filename, str), "filename must be a string"
        df.to_csv(filename, index=False)

    def get_raw_data(self):
        """Returns the raw dataset."""
        assert self.df is not None, "Raw data not loaded. Run download_data() first."
        return self.df

