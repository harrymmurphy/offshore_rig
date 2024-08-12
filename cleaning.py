# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:23:26 2024

@author: Harry Murphy
"""
import pandas as pd
import cpi
import re

# Define the file path
file_path = r'C:\Users\Harry Murphy\OneDrive\Desktop\drillship_timeseries_edit.csv'

# Load the dataset from the specified file
df = pd.read_csv(file_path)

# Function to clean column names
def clean_column_name(name):
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r'[^\w]', '_', name)
    # Replace double underscores with single underscores
    name = re.sub(r'_{2,}', '_', name)
    # Remove any leading or trailing underscores
    name = name.strip('_')
    return name

# Clean the column names
df.columns = [clean_column_name(col) for col in df.columns]

# Check names for use in analysis
print("Cleaned Column Names:")
for column in df.columns:
    print(column)

# Function to convert monetary values and percentages
def convert_entry(value):
    if isinstance(value, str):
        # Handle dollar signs and commas
        if '$' in value:
            value = re.sub(r'[,$]', '', value)  # Remove dollar signs and commas
            return float(value)  # Convert to float
        # Handle percent signs
        elif '%' in value:
            value = re.sub(r'[%]', '', value)  # Remove percent signs
            return float(value) / 100  # Convert to decimal
        # Handle empty strings and other cases
        elif value.strip() == '':
            return None  # Use None for empty entries
        else:
            try:
                return float(value)  # Convert to float
            except ValueError:
                return value  # Return original value if conversion fails
    return value

# Apply the conversion to the entire DataFrame
for column in df.columns:
    df[column] = df[column].apply(convert_entry)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Print the DataFrame after date parsing
print("DataFrame After Date Parsing:")
print(df.head())

# Define date ranges for filtering
adjustment_cutoff = pd.Timestamp('2024-01-01')
pre_adjustment_df = df[df['Date'] < adjustment_cutoff]
post_adjustment_df = df[df['Date'] >= adjustment_cutoff]

# Extract year from the 'Date' column for CPI adjustment in the pre_adjustment DataFrame
pre_adjustment_df['release_year'] = pre_adjustment_df['Date'].dt.year

# Function to adjust for inflation using CPI
def adjust_for_inflation(value, release_year, target_year):
    if pd.notna(value) and value > 0:
        try:
            # Use the last available year for adjustment
            if target_year > release_year:
                return cpi.inflate(value, release_year, target_year)
            else:
                return value
        except Exception as e:
            print(f"Error adjusting value {value} from {release_year} to {target_year}: {e}")
            return value  # Return original value if CPI data is not available or an error occurs
    return value

# Adjust inflation for all price columns in the pre_adjustment DataFrame
price_columns = [
    'Min_Lead_Edge_Dayrate', 
    'Avg_Lead_Edge_Dayrate', 
    'Max_Lead_Edge_Dayrate', 
    'Crude_Price',
    'Deepwater_Investments'
]

target_year = 2023  # Use the last available year for inflation adjustment

for column in price_columns:
    if column in pre_adjustment_df.columns:
        pre_adjustment_df[column] = pre_adjustment_df.apply(lambda row: adjust_for_inflation(row[column], row['release_year'], target_year), axis=1)
    else:
        print(f"Column '{column}' not found in DataFrame!")

# Drop the 'release_year' column from the pre_adjustment DataFrame
pre_adjustment_df = pre_adjustment_df.drop(columns=['release_year'])

# Combine the pre_adjustment and post_adjustment DataFrames
final_df = pd.concat([pre_adjustment_df, post_adjustment_df])

# Print the cleaned and adjusted DataFrame
print("Final Cleaned and Adjusted DataFrame:")
print(final_df)

# Save to csv

output_file_path = r'C:\Users\Harry Murphy\OneDrive\Desktop\drillship_timeseries_adjusted.csv'
final_df.to_csv(output_file_path, index=False)
print(f"Final DataFrame has been saved to {output_file_path}")
