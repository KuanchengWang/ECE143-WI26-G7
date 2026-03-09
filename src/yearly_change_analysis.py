import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import constants

class YearlyChangeAnalysis:
    def __init__(self, df) -> None:
        self.df = df if df is not None else pd.read_csv(constants.DATASET_PATH)
        self.delay_min_cols = ['carrier_delay','weather_delay','nas_delay','security_delay','late_aircraft_delay']
        self.cause_labels = ['Carrier','Weather','NAS','Security','Late Aircraft']

    def get_yearly_df(self):
        yearly = self.df.groupby('year')[self.delay_min_cols].sum() / 1e6
        return yearly
    
    def describe(self):
        yearly = self.get_yearly_df()
        return yearly.describe().round(2)

    def plot_change_by_cause(self):
        yearly = self.get_yearly_df()
        yearly = self.get_yearly_df()
        figure, axis = plt.subplots(figsize= (13, 5))
        for i, j in enumerate[str](self.delay_min_cols):
            label = self.cause_labels[self.delay_min_cols.index(j)]
            axis.plot(yearly.index.astype(int),
                      yearly[j],
                      marker='o',
                      color = constants.PALETTE[i],
                      label = label,)
        axis.set_title('Year-by-Year improvement/changes in each factor of overall delay', fontsize=13, fontweight='bold')
        axis.set_ylabel('Delay')
        axis.set_xlabel('Year')
        axis.legend(title='Cause')
        axis.set_xticks(yearly.index.astype(int))
        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'yearlychange.png')
        plt.show()

    def plot_change_percent_by_cause(self):
        #quantifying the change for each cause year-to-year
        yearly = self.get_yearly_df()
        percent = yearly.pct_change() * 100
        percent = percent.dropna()

        fig2, axis2 = plt.subplots(figsize = (13, 5))
        for i, j in enumerate[str](self.delay_min_cols):
            label = self.cause_labels[self.delay_min_cols.index(j)]
            axis2.plot(percent.index.astype(int),
                        percent[j],
                        marker = 'o',
                        color = constants.PALETTE[i % len(constants.PALETTE)],
                        label = label,
                        )
        axis2.axhline(0, color='blue', linestyle='--', alpha=0.6)
        axis2.set_title('Year to year change in each cause', fontsize=13, fontweight='bold')
        axis2.set_ylabel('Percent change vs prior year')
        axis2.set_xlabel('Year')
        axis2.legend(title='Cause')
        axis2.set_xticks(percent.index.astype(int))
        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'yeartoyearpctchange.png')
        plt.show()
