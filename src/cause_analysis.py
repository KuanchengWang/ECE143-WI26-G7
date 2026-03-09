import pandas as pd
import matplotlib.pyplot as plt

from src import constants


class DelayCauseAnalysis:
    """Analyze and visualize the distribution of flight delay causes.
    Attributes:
        df (pd.DataFrame): DataFrame containing flight delay data.
        delay_cause_cols (list): List of columns representing different delay causes.
        cause_counts (pd.Series): Total count of delay incidents for each cause.
        cause_minutes (pd.Series): Total delay minutes attributed to each cause.
        cause_colors (dict): Mapping of delay causes to specific colors for visualization.
    """
    
    def __init__(self, df = None):
        """Initialize the DelayCauseAnalysis object.
        Args:
            df (pd.DataFrame): DataFrame containing flight delay data. If None, it will be loaded from the dataset path.
        """
        self.df = df if df is not None else pd.read_csv(constants.DATASET_PATH)
        
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
        """Provide a statistical summary of the delay cause data.
        Returns:
            pd.DataFrame: DataFrame containing the statistical summary for the delay cause metrics.
        """
        subset = self.df[self.delay_cause_cols]
        assert not subset.empty, "No delay cause columns available for description"
        return subset.describe().round(2)
    
    def plot_incidents_and_minutes(self):
        """Plot side-by-side pie charts showing the distribution of delay incidents and total delay minutes by cause."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        assert not self.cause_counts.empty and not self.cause_minutes.empty, "Cause counts or minutes are empty, cannot plot"
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
        """Plot a horizontal bar chart showing the average delay duration per delayed flight for each cause."""
        avg_per_cause = (self.cause_minutes / self.cause_counts).sort_values(ascending=True) # type: ignore
        assert not avg_per_cause.empty, "Average per cause is empty, cannot plot."
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
        ax.set_title('Yearly Average Minutes per Delayed Flight (2013-23)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, avg_per_cause.max() * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(constants.SAVE_DIR + 'DelayCauseAnalysis_plot_avg_delay_per_delayed_flight.png')
        plt.show()