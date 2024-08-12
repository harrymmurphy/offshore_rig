import pandas as pd
import matplotlib.pyplot as plt

# File paths and chart names
file_paths = {
    '5G_6G_drillship': r'C:\Users\Harry Murphy\OneDrive\Desktop\drillship_data_full.csv',
    'jackups': r'C:\Users\Harry Murphy\OneDrive\Desktop\jackups_full.csv',
    'Benign_Semis_and_Low_Spec_Drillships': r'C:\Users\Harry Murphy\OneDrive\Desktop\Benign_Semis_and_Low_Spec_Rigs.csv',
    '7G_8G_drillship': r'C:\Users\Harry Murphy\OneDrive\Desktop\7G_8G_drillship_full.csv'
}

# Corresponding chart names
chart_names = {
    '5G_6G_drillship': '5G and 6G Drillships',
    'jackups': 'Jackups',
    'Benign_Semis_and_Low_Spec_Drillships': 'Benign Environment Semis and Low-Spec Drillships',
    '7G_8G_drillship': '7G and 8G Drillships'
}

# Define colors for each dataset
colors = {
    '5G_6G_drillship': 'blue',
    'jackups': 'green',
    'Benign_Semis_and_Low_Spec_Drillships': 'orange',
    '7G_8G_drillship': 'purple'
}

# Define the zoom range and restriction range
zoom_start = pd.Timestamp('2015-01-01')
zoom_end = pd.Timestamp('2029-12-01')
restriction_start = pd.Timestamp('2010-01-01')
restriction_end = pd.Timestamp('2029-07-01')

# Function to process data
def process_data(file_path, dataset_name):
    # Load the data from the CSV file
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
    
    # Strip any extra whitespace from column names
    data.columns = data.columns.str.strip()
    
    # Ensure 'Date' column is in datetime format and strip time component
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Date'] = data['Date'].dt.normalize()
    
    # Define the date adjustment
    def adjust_date(date):
        if date.day > 15:
            return (date.replace(day=1) + pd.DateOffset(months=1)).normalize()
        else:
            return date.replace(day=1).normalize()
    
    # Apply the adjustment function to the 'Date' column
    data['AdjustedDate'] = data['Date'].apply(adjust_date)
    
    # Restrict the dataset to the specified range
    data = data[(data['AdjustedDate'] >= restriction_start) & (data['AdjustedDate'] <= restriction_end)]
    
    # Check if 'Oil' column exists and contains valid numeric data
    if 'Oil' not in data.columns:
        raise ValueError(f"The 'Oil' column was not found in the data for {dataset_name}.")
    data['Oil'] = pd.to_numeric(data['Oil'], errors='coerce')
    data = data.dropna(subset=['Oil'])
    
    # Group by the adjusted date and find the maximum oil value for each group
    result = data.groupby('AdjustedDate')['Oil'].max().reset_index()
    
    # Rename columns
    result.columns = ['Date', f'{dataset_name}_Oil_Production']
    
    # Remove duplicates and keep the maximum value
    result = result.groupby('Date').max().reset_index()
    
    # Sort the results by date
    result = result.sort_values('Date')
    
    # Set the 'Date' column as the index
    result.set_index('Date', inplace=True)
    
    # Create a complete date range with monthly frequency within the restriction range
    full_range = pd.date_range(start=restriction_start, end=restriction_end, freq='MS')
    
    # Reindex the DataFrame to include all months in the full range, setting missing months to NaN
    result = result.reindex(full_range)
    
    # Interpolate missing values using linear interpolation
    result = result.interpolate(method='linear')
    
    # Convert the DataFrame to a time series
    time_series = pd.Series(result[f'{dataset_name}_Oil_Production'], name=f'{dataset_name}_Oil_Production')
    
    # Calculate the rate of change (percentage change)
    rate_of_change = time_series.pct_change() * 100
    
    # Drop NaN values in the rate of change series
    rate_of_change = rate_of_change.dropna()
    
    return time_series, rate_of_change

# Process each dataset
time_series_data = {}
rate_of_change_data = {}

for dataset_name in file_paths.keys():
    time_series, rate_of_change = process_data(file_paths[dataset_name], dataset_name)
    time_series_data[dataset_name] = time_series
    rate_of_change_data[dataset_name] = rate_of_change

# Calculate individual rig types
jackups = time_series_data['jackups']
semisubmersibles_and_low_spec = time_series_data['Benign_Semis_and_Low_Spec_Drillships'] - jackups
five_and_six_g_drillships = time_series_data['5G_6G_drillship'] - semisubmersibles_and_low_spec - jackups
seven_and_eight_g_drillships = time_series_data['7G_8G_drillship'] - semisubmersibles_and_low_spec - five_and_six_g_drillships - jackups

# Define names and colors for individual rigs
individual_rigs = {
    'Jackups': jackups,
    'Semisubmersibles and Low-Spec Drillships': semisubmersibles_and_low_spec,
    '5G and 6G Drillships': five_and_six_g_drillships,
    '7G and 8G Drillships': seven_and_eight_g_drillships
}

# Define colors for the plots
plot_colors = {
    'Jackups': 'green',
    'Semisubmersibles and Low-Spec Drillships': 'orange',
    '5G and 6G Drillships': 'blue',
    '7G and 8G Drillships': 'purple'
}

# Initialize plot
plt.figure(figsize=(14, 10))

for i, (name, time_series) in enumerate(individual_rigs.items()):
    # Calculate rate of change for individual rig
    rate_of_change = time_series.pct_change() * 100
    rate_of_change = rate_of_change.dropna()
    
    # Apply exponential smoothing to the rate of change
    rate_of_change_smoothed = rate_of_change.ewm(span=12, adjust=False).mean()
    
    # Apply rolling median filter to smooth out outliers
    rolling_median = rate_of_change_smoothed.rolling(window=12, center=True, min_periods=1).median()
    rate_of_change_smoothed = rolling_median
    
    # Plot the time series with markers
    plt.subplot(4, 2, i * 2 + 1)
    plt.plot(time_series.index, time_series, label=f'{name} Oil Production', color=plot_colors[name], marker='o', linestyle='-')
    plt.title(f'{name} Oil Production')
    plt.xlabel('Date')
    plt.ylabel('Oil Production')
    plt.legend()
    plt.grid(True)
    plt.xlim(zoom_start, zoom_end)
    plt.ylim(0, 7.5)  # Set y-axis limit for Oil Production

    # Plot the rate of change with markers
    plt.subplot(4, 2, i * 2 + 2)
    plt.plot(rate_of_change_smoothed.index, rate_of_change_smoothed, label=f'{name} Rate of Change (Smoothed)', color=plot_colors[name], linestyle='-')
    plt.title(f'{name} Rate of Change')
    plt.xlabel('Date')
    plt.ylabel('Rate of Change (%)')
    plt.legend()
    plt.grid(True)
    plt.xlim(zoom_start, zoom_end)
    plt.ylim(-2, 4)  # Set y-axis limit for Rate of Change

# Adjust layout and show plot
plt.tight_layout()

# Save the plot as a file
plot_file_path = r'C:\Users\Harry Murphy\OneDrive\Desktop\individual_rigs_oil_production_and_rate_of_change_zoomed_plot.png'
plt.savefig(plot_file_path)

print(f"Plot saved to '{plot_file_path}'.")
