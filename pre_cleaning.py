import pandas as pd

# Path to the CSV file
file_path = r'C:\Users\Harry Murphy\OneDrive\Desktop\Westwood_Rig_data.csv'

# Load the CSV file
df_raw = pd.read_csv(file_path)

# Display the first few rows to understand its structure
print(df_raw.head())

# Define function to clean monetary values and percentages
def clean_value(value):
    if isinstance(value, str):
        value = value.replace('$', '').replace(',', '').replace('%', '')
        try:
            return float(value)
        except ValueError:
            return None
    return value

# Create empty DataFrames for each type
drillship_df = pd.DataFrame()
jackup_df = pd.DataFrame()
semisub_df = pd.DataFrame()

# Define metrics to include
metrics = ['Min. Lead. Edge Dayrate', 'Avg. Lead. Edge Dayrate', 
           'Max. Lead. Edge Dayrate', 'Rigs Working', 'Utilization (MF)', 
           'Rigs Committed', 'Marketed Supply', 'Comm. Utilization (MF)', 'Total Supply']

# Get unique years from the data
years = df_raw.columns[df_raw.columns.str.contains(r'\d{4}')].str.extract(r'(\d{4})')[0].unique()

# Iterate over each year
for year in years:
    for metric in metrics:
        for month in range(1, 13):  # Iterate from January to December
            for type_ in ['Drillship', 'Jackup', 'Semisub']:
                col_name = f'{year} {pd.to_datetime(month, format="%m").strftime("%b")} {type_}'
                if col_name in df_raw.columns:
                    value = df_raw[df_raw['Metric'] == metric][col_name].values
                    if value.size > 0:
                        value = clean_value(value[0])
                        date = pd.to_datetime(f'{year}-{month:02d}-01', format='%Y-%m-%d')
                        
                        if type_ == 'Drillship':
                            drillship_df.loc[date, metric] = value
                        elif type_ == 'Jackup':
                            jackup_df.loc[date, metric] = value
                        elif type_ == 'Semisub':
                            semisub_df.loc[date, metric] = value

# Clean and format DataFrames
def clean_dataframe(df):
    df = df.applymap(clean_value)
    return df

drillship_df = clean_dataframe(drillship_df)
jackup_df = clean_dataframe(jackup_df)
semisub_df = clean_dataframe(semisub_df)

# Save the cleaned DataFrames to new CSV files
drillship_df.to_csv(r'C:\Users\Harry Murphy\OneDrive\Desktop\cleaned_drillship_data.csv')
jackup_df.to_csv(r'C:\Users\Harry Murphy\OneDrive\Desktop\cleaned_jackup_data.csv')
semisub_df.to_csv(r'C:\Users\Harry Murphy\OneDrive\Desktop\cleaned_semisub_data.csv')

# Display the cleaned DataFrames
print("Drillship DataFrame:")
print(drillship_df)
print("\nJackup DataFrame:")
print(jackup_df)
print("\nSemisub DataFrame:")
print(semisub_df)
