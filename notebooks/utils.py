import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


def display_hierarchical_timeseries(y_train, y_test, forecasts = None):
    
    y_column_name = y_train.columns[0]
    # Function to plot based on the selected state
    def plot_for_given_level(selected_first_level):
        second_level_values = y_train.loc[selected_first_level].index.get_level_values(0).unique()
        
        # Create subplots
        fig, axs = plt.subplots(nrows=int(np.ceil(len(second_level_values) / 2)),
                                ncols=2,
                                figsize=(12, 1.5 * len(second_level_values)))
        
        y_train_selected = y_train.loc[selected_first_level]
        y_test_selected = y_test.loc[selected_first_level]
        
        if forecasts is not None:
            forecasts_selected = {k: forecast.loc[selected_first_level] for k, forecast in forecasts.items()}
            
        
        # Plot each purpose and remove empty subplots in one loop
        for ax, level_value in zip(axs.flatten(), second_level_values):
            y_train_selected.loc[level_value][y_column_name].rename("Train").plot(ax=ax, legend=True)
            y_test_selected.loc[level_value][y_column_name].rename("Test").plot(ax=ax, legend=True)
            
            if forecasts is not None:
                for name, forecast in forecasts_selected.items():
                    forecast.loc[level_value][y_column_name].rename(name).plot(ax=ax, legend=True)
                    
            ax.set_title(level_value)
            
        # Remove empty subplots
        for ax in axs.flatten()[len(second_level_values):]:
            fig.delaxes(ax)
        
        fig.tight_layout()
        plt.show()

    # Get the list of available values at level 0
    available_first_level = y_train.index \
                        .get_level_values(0) \
                        .unique()
                        

    # Create a dropdown widget for series selection
    first_level_selector = widgets.Dropdown(
        options=[option for option in available_first_level if option != "__total"],
        value=available_first_level[0],
        description='Level 0:',
        disabled=False,
    )

    # Display the widget and link it to the plotting function
    return widgets.interactive(plot_for_given_level, selected_first_level=first_level_selector)