import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from src import constants

class AirportDelayAnalysis:
    """Analyze and visualize delay patterns across airports.
    Attributes:
        df (pd.DataFrame): The DataFrame containing flight delay data.
        agg_cols_dict (dict): Defines the aggregation operations for delay metrics.
        cause_cols (list): Represents different delay causes.
        cause_lables (dict): Mapping of delay cause columns to their human-readable labels.
    """

    def __init__(self, df = None):
        """Initialize the AirportDelayAnalysis object.
        Args:
            df (pd.DataFrame): DataFrame containing flight delay data. If None, it will be loaded from the dataset path.
        """
        self.df = df if df is not None else pd.read_csv(constants.DATASET_PATH)
        self.agg_cols_dict = {
            'total_flights': ('arr_flights', 'sum'),
            'total_del15': ('arr_del15', 'sum'),
            'total_delay_m': ('arr_delay', 'sum'),
            'weather_m': ('weather_delay', 'sum'),
            'carrier_m': ('carrier_delay', 'sum'),
            'security_m': ('security_delay', 'sum'),
            'late_ac_m': ('late_aircraft_delay', 'sum'),
            'nas_m': ('nas_delay', 'sum'),
        }
        self.cause_cols   = ['carrier_m', 'weather_m', 'nas_m', 'security_m', 'late_ac_m']
        self.cause_lables  = {
            'carrier_m': 'Carrier', 
            'weather_m': 'Weather',
            'nas_m': 'NAS',
            'security_m': 'Security',
            'late_ac_m': 'Late Aircraft'
        }

    def get_airport_df(self):
        """Aggregate the flight delay data by airport.
        Returns:
            pd.DataFrame: DataFrame containing aggregated delay metrics for each airport.
        """
        ap_agg = self.df.groupby(['airport', 'airport_name']).agg(**self.agg_cols_dict).reset_index()
        ap_agg['delay_rate']    = ap_agg['total_del15']  / ap_agg['total_flights'] * 100
        ap_agg['avg_delay_min'] = ap_agg['total_delay_m'] / ap_agg['total_del15'].replace(0, np.nan)
        return ap_agg
    
    def describe(self):
        """Provide a statistical summary of the aggregated airport delay data.
        Returns:
            pd.DataFrame: DataFrame containing the statistical summary for the airport delay metrics.
        """
        ap = self.get_airport_df()
        assert not ap.empty, "Airport DataFrame is empty, cannot describe"
        return ap.describe().round(2)
    
    def get_top(self, col, n = 20):
        """
        Get the top n airports based on a specified column (eg, total_flights, delay_rate).
        Args:            
            col (str): Column name to sort by (eg, 'total_flights', 'delay_rate').
            n (int): Number of top airports to return.
        Returns:
            pd.DataFrame: DataFrame containing the top n airports based on the specified column.
        """
        ap = self.get_airport_df()
        assert isinstance(col, str), "col must be a string"
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        assert col in ap.columns, f"Column '{col}' not found in airport DataFrame."
        top_n = ap.nlargest(n, col).sort_values(col, ascending=True)
        return top_n        

    def get_short_names(self, row):
        """
        Generate a short name for the airport, combines airport code and truncated airport name.
        Args:
            row (pd.Series): Row from the airport DataFrame.
        Returns:
            str: Short name for the airport, format "CODE — Name".
        """
        assert hasattr(row, 'airport') and hasattr(row, 'airport_name'), "row must have 'airport' and 'airport_name' attributes"
        return f"{row.airport} — {row.airport_name.split(':')[0].strip()[:25]}"

    def plot_top_airports_bar(self, col='total_flights', n=20):
        """Plot horizontal bar chart of the top n airports based on a specified column.
        Args:
            col (str): Column name to sort by for selecting top airports (e.g., 'total_flights', 'delay_rate').
            n (int): Number of top airports to include in the plot.
        """
        ap_top_n = self.get_top(col, n)
        assert not ap_top_n.empty, f"No data available to plot, empty for top {n}."
        labels = [self.get_short_names(row) for row in ap_top_n.itertuples()]
        ap_top_n['dominant_cause'] = ap_top_n[self.cause_cols].idxmax(axis=1).map(self.cause_lables)
        
        cmap = plt.cm.get_cmap('Oranges')
        norm = Normalize(ap_top_n['avg_delay_min'].min(), ap_top_n['avg_delay_min'].max())
        colors = [cmap(norm(v)) for v in ap_top_n['avg_delay_min']]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(labels, 
                       ap_top_n['delay_rate'].values,  # type: ignore
                       color = colors,)

        for bar, (_, row) in zip(bars, ap_top_n.iterrows()):
            w = bar.get_width()
            ax.text(w + 0.2, 
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}% | {row['dominant_cause']} | {row['total_flights']/1e6:.1f}M flights",
                    fontsize=8,)

        mean_line = ap_top_n['delay_rate'].mean()
        ax.axvline(mean_line, linestyle='--', label=f'Mean delay rate: {mean_line:.1f}%', color='grey', alpha=0.5)
        ax.set_title('Top 20 Airports by Volume (Bar-Colour = Avg Delay Duration)\n'
                     'Annotation = delay rate | dominant cause | total volume',
                     fontsize=12, fontweight='bold')

        ax.legend(fontsize=8, loc='lower right')
        # plt.colorbar(plt.cm.Oranges, ax=ax, label='Avg delay (min)')
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label='Avg delay (min)')
        
        ax.set_xlabel('Delay Rate — flights delayed ≥ 15 min (%)')
        ax.set_xlim(0, ap_top_n['delay_rate'].max() * 1.4)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'AirportDelayAnalysis_plot_top_airports_bar.png')
        plt.show()