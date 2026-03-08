import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import constants


class CarrierDelayAnalysis:
    def __init__(self, csv_file = constants.DATASET_PATH) -> None:
        self.df = pd.read_csv(csv_file)
        self.agg_cols_dict = {
            'total_flights': ('arr_flights', 'sum'),
            'total_delayed': ('arr_del15', 'sum'),
            'total_delay_min': ('late_aircraft_delay', 'sum'),
            'cancelled': ('arr_cancelled', 'sum'),
        }

    def get_carrier_df(self):
        # Create aggregator for Carrier
        ca_agg = self.df.groupby('carrier_name').agg(**self.agg_cols_dict).reset_index()
        ca_agg['delay_rate'] = ca_agg['total_delayed'] / ca_agg['total_flights'] * 100
        ca_agg['avg_delay_min'] = ca_agg['total_delay_min'] / ca_agg['total_delayed']
        ca_agg['cancel_rate'] = ca_agg['cancelled'] / ca_agg['total_flights'] * 100
        ca_agg['short_name'] = ca_agg['carrier_name'].apply(lambda x: x[:10])
        ca_agg['bubble_size'] = (ca_agg['total_flights'] / ca_agg['total_flights'].max()) * 1e4
        return ca_agg

    def describe(self):
        ca = self.get_carrier_df()
        return ca.describe().round(2)

    def plot_carrier_profile_bubble(self):
        carrier_df = self.get_carrier_df()
        fig, ax = plt.subplots(figsize=(12, 8))
        # 1. Darker color - higher cancellation rate
        # 2. Bubble Size - Volume the carrier handles
        # 3. Towards left - less delay rate
        # 4. Towards bottom - less delay duration
        sc = ax.scatter(carrier_df['delay_rate'], 
                        carrier_df['avg_delay_min'],
                        s = carrier_df['bubble_size'], 
                        c = carrier_df['cancel_rate'], 
                        cmap = 'Oranges',
                        alpha = 0.7,)
        plt.colorbar(sc, ax=ax, label='Cancellation Rate (%)')

        for _, row in carrier_df.iterrows():
            ax.annotate(row['short_name'], 
                        (row['delay_rate'], row['avg_delay_min']),
                        ha='center',)

        ax.set_xlabel('Delay Rate (%)')
        ax.set_ylabel('Avg Delay Duration (min)')
        ax.set_title('Carrier Delay Profile - Rate vs Duration (2013-23)\n(bubble = volume, colour = cancel rate)', fontweight='bold')
        ax.axhline( carrier_df['avg_delay_min'].median(), color='grey', linestyle='--', alpha=0.7)
        ax.axvline(carrier_df['delay_rate'].median(), color='grey', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'CarrierDelayAnalysis_plot_carrier_profile_bubble.png')
        plt.show()
