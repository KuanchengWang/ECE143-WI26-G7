import pandas as pd
import kagglehub
from pathlib import Path
from typing import Optional

from src import constants

class Preprocessing:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.delay_cols = ['arr_delay', 
                           'carrier_delay', 
                           'weather_delay', 
                           'nas_delay', 
                           'security_delay', 
                           'late_aircraft_delay']
        
    def download_data(self):
        try:
            path = kagglehub.dataset_download(constants.DATASET_HANDLE)
            dataset_dir = Path(path)
            
            csv_files = list(dataset_dir.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError("No CSV file found in the downloaded dataset folder.")
            
            file_path = csv_files[0]
            self.df = pd.read_csv(file_path)
            
        except Exception as e:
            raise Exception(f"Failed to download or load data: {e}")

    def clean_data(self):
        if self.df is None:
            raise ValueError("Dataframe is empty. Run load_data() first.")
        
        df_clean = self.df.copy()

        missing = df_clean.isnull().sum()
        missing_pct = (missing / len(df_clean) * 100).round(2)
        missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        df_clean = df_clean.dropna(subset = self.delay_cols).copy()

    def save_data(self, filename = constants.DATASET_PATH):
        assert self.df is not None, "Dataframe is empty. Run load_data() and clean_data() first."
        self.df.to_csv(filename, index=False)

