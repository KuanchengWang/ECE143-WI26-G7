import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import constants


class MonthlyDelayAnalysis:
    def __init__(self, csv_file = constants.DATASET_PATH) -> None:
        self.df = pd.read_csv(csv_file)
        self.month_map = {
            1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun', 
            7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'
        }
        self.cause_colors = {
            'Carrier': constants.PALETTE[0],
            'Weather': constants.PALETTE[1],
            'NAS': constants.PALETTE[2],
            'Security': constants.PALETTE[3],
            'Late Aircraft': constants.PALETTE[4],
        }
        self.cause_cols   = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
        self.count_cols   = ['carrier_ct',    'weather_ct',    'nas_ct',    'security_ct',    'late_aircraft_ct']
        self.cause_labels = ['Carrier', 'Weather', 'NAS', 'Security', 'Late Aircraft']

    def get_monthly_df(self):
        monthly = self.df.groupby('month').agg(
            flights = ('arr_flights', 'mean'),
            del15 = ('arr_del15', 'mean'),
            carrier = ('carrier_ct', 'mean'),
            weather = ('weather_ct', 'mean'),
            nas = ('nas_ct', 'mean'),
            security = ('security_ct', 'mean'),
            late_ac = ('late_aircraft_ct', 'mean')
        ).reset_index()
        monthly['delay_rate'] = monthly['del15'] / monthly['flights'] * 100
        monthly['month_name'] = monthly['month'].map(self.month_map)
        return monthly
    
    def describe(self):
        monthly = self.get_monthly_df()
        return monthly.describe().round(2)
    
    def plot_delay_rate(self):
        monthly = self.get_monthly_df()
        fig, ax = plt.subplots(figsize=(12, 5))
        # delay rate line
        ax.plot(monthly['month_name'], 
                monthly['delay_rate'],
                color = constants.PALETTE[0], 
                marker = 'o')
        ax.fill_between(monthly['month_name'], 
                        monthly['delay_rate'],
                        alpha = 0.1, 
                        color = constants.PALETTE[0])
        ax.set_title('Monthly Delay Rate (%) — Seasonal Effect', fontsize=14, fontweight='bold')
        ax.set_ylabel('Delay Rate (%)')
        ax.set_ylim(0, monthly['delay_rate'].max() * 1.2)
        for x, y in zip(monthly['month_name'], monthly['delay_rate']):
            ax.annotate(f'{y:.1f}%', (x, y), 
                        textcoords='offset points',
                        xytext=(0, 8), 
                        ha='center')

        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'MonthlyDelayAnalysis_plot_delay_rate.png')
        plt.show()

    def plot_delay_incidents(self):
        monthly = self.get_monthly_df()
        # delay counts bar
        fig, ax = plt.subplots(figsize=(12, 5))
        cause_cols = ['carrier','weather','nas','security','late_ac']
        cause_labels = ['Carrier','Weather','NAS','Security','Late Aircraft']
        bottom = np.zeros(len(monthly))
        for col, label in zip(cause_cols, cause_labels):
            ax.bar(monthly['month_name'], 
                   monthly[col], 
                   bottom = bottom,
                   label = label, 
                   color = self.cause_colors[label])
            bottom += monthly[col].values # type: ignore
        ax.set_title('Monthly Delay Incidents by Cause', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Delay Incidents')
        ax.legend(loc='upper left', fontsize=8, ncol=5)

        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'MonthlyDelayAnalysis_plot_delay_incidents.png')
        plt.show()

    def plot_avg_duration_per_delayed_flight(self):
        # avg delay minutes bar
        # aggregate monthly with minute totals
        monthly_min = self.df.groupby('month').agg(
            carrier_delay = ('carrier_delay', 'sum'),
            weather_delay = ('weather_delay', 'sum'),
            nas_delay = ('nas_delay', 'sum'),
            security_delay = ('security_delay', 'sum'),
            late_aircraft_delay = ('late_aircraft_delay', 'sum'),
            carrier_ct = ('carrier_ct', 'sum'),
            weather_ct = ('weather_ct', 'sum'),
            nas_ct = ('nas_ct', 'sum'),
            security_ct = ('security_ct', 'sum'),
            late_aircraft_ct = ('late_aircraft_ct', 'sum'),
            year_min = ('year', 'min'),
            year_max = ('year', 'max'),
        ).reset_index()
        monthly_min['month_name'] = monthly_min['month'].map(self.month_map)

        # compute avg minutes per incident for each cause
        year_count = monthly_min['year_max'] - monthly_min['year_min'] + 1
        for min_col, ct_col in zip(self.cause_cols, self.count_cols):
            monthly_min[min_col + '_avg'] = monthly_min[min_col] / (monthly_min[ct_col].replace(0, np.nan) * year_count)

        avg_cols = [c + '_avg' for c in self.cause_cols]

        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = np.zeros(len(monthly_min))
        for col, label in zip(avg_cols, self.cause_labels):
            ax.bar(monthly_min['month_name'], 
                   monthly_min[col],
                   bottom = bottom,
                   label = label,
                   color = self.cause_colors[label])
            bottom += monthly_min[col].fillna(0).values # type: ignore

        ax.set_title('Monthly Avg. Delay Duration by Cause', fontsize=14, fontweight='bold')
        ax.set_ylabel('Avg. Delay (minutes per incident)')
        ax.legend(loc='upper left', fontsize=8, ncol=5)

        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'MonthlyDelayAnalysis_plot_avg_duration_per_delayed_flight.png')
        plt.show()