import pandas as pd
import matplotlib.pyplot as plt

from src import constants


class DelayCauseAnalysis:
    def __init__(self, csv_file = constants.DATASET_PATH) -> None:
        self.df = pd.read_csv(csv_file)
        
        self.delay_cause_cols = ['carrier_ct','weather_ct','nas_ct','security_ct','late_aircraft_ct']
        
        self.cause_counts = pd.Series({
            'Carrier': self.df['carrier_ct'].sum(),
            'Weather': self.df['weather_ct'].sum(),
            'NAS': self.df['nas_ct'].sum(),
            'Security': self.df['security_ct'].sum(),
            'Late Aircraft': self.df['late_aircraft_ct'].sum()
        })

        self.cause_minutes = pd.Series({
            'Carrier': self.df['carrier_delay'].sum(),
            'Weather': self.df['weather_delay'].sum(),
            'NAS': self.df['nas_delay'].sum(),
            'Security': self.df['security_delay'].sum(),
            'Late Aircraft': self.df['late_aircraft_delay'].sum()
        })

        self.cause_colors = {
        'Carrier': constants.PALETTE[0],
        'Weather': constants.PALETTE[1],
        'NAS': constants.PALETTE[2],
        'Security': constants.PALETTE[3],
        'Late Aircraft': constants.PALETTE[4],
    }

    def describe(self):
        return self.df[self.delay_cause_cols].describe().round(2)
    
    def plot_incidents_and_minutes(self):
        # Overall Cause Distribution
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Flight Delay — Distribution by Cause (2013-23)', fontsize=14, fontweight='bold')

        sorted_cause_counts = self.cause_counts.sort_values(ascending=False)
        sorted_cause_minutes = self.cause_minutes[sorted_cause_counts.index]    

        # Pie (Incidents)
        colors=[ self.cause_colors[c] for c in sorted_cause_counts.index ]
        ax[0].pie(sorted_cause_counts, 
                  labels = sorted_cause_counts.index, 
                  autopct = '%1.1f%%', 
                  colors = colors,
                  explode = [0.05]*5,
                  wedgeprops = {'edgecolor':'white','linewidth':1.5})
        ax[0].set_title('Share of Delay Incidents')

        # Pie (Minutes)
        colors=[ self.cause_colors[c] for c in sorted_cause_minutes.index ]
        ax[1].pie(sorted_cause_minutes, 
                  labels = sorted_cause_minutes.index, 
                  autopct = '%1.1f%%', 
                  colors = colors,
                  explode = [0.05]*5,
                  wedgeprops = {'edgecolor':'white','linewidth':1.5})
        ax[1].set_title('Share of Total Delay Minutes')
            
        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'DelayCauseAnalysis_plot_incidents_and_minutes.png')
        plt.show()

    def plot_avg_delay_per_delayed_flight(self):
        # Bar
        avg_per_cause = (self.cause_minutes / self.cause_counts).sort_values(ascending=True) # type: ignore
        color=[ self.cause_colors[c] for c in avg_per_cause.index ]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(avg_per_cause.index, 
                       avg_per_cause.values, 
                       color = color)

        for bar in bars:
            ax.text(bar.get_width() + 0.5, 
                    bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.1f} min', 
                    va='center',
                    fontweight='bold')

        ax.set_xlabel('Average Delay Duration per Incident (minutes)')
        ax.set_title('Average Minutes per Delayed Flight', fontsize=14, fontweight='bold')
        ax.set_xlim(0, avg_per_cause.max() * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'DelayCauseAnalysis_plot_avg_delay_per_delayed_flight.png')
        plt.show()