import pandas as pd
import numpy as np
raw_df = ('your file')
def process_dataset(df):
    # Step 1: Calculate Supply Tightness Index
    
    # Calculate Utilization and Committed Utilization
    df['Utilization'] = df['Rigs Working'] / df['Total Supply']
    df['Committed Utilization'] = df['Rigs Committed'] / df['Total Supply']
    
    # Normalize Utilization and Committed Utilization
    min_utilization = df['Utilization'].min()
    max_utilization = df['Utilization'].max()
    df['Normalized Utilization'] = (df['Utilization'] - min_utilization) / (max_utilization - min_utilization)
    
    min_committed_utilization = df['Committed Utilization'].min()
    max_committed_utilization = df['Committed Utilization'].max()
    df['Normalized Committed Utilization'] = (df['Committed Utilization'] - min_committed_utilization) / (max_committed_utilization - min_committed_utilization)
    
    # Define weight parameter (alpha) for the index
    alpha = 0.5
    
    # Calculate Supply Tightness Index
    df['Supply Tightness Index'] = alpha * df['Normalized Utilization'] + (1 - alpha) * df['Normalized Committed Utilization']
    
    # Scale to range [-1, 1]
    min_index = df['Supply Tightness Index'].min()
    max_index = df['Supply Tightness Index'].max()
    df['Scaled Supply Tightness Index'] = 2 * ((df['Supply Tightness Index'] - min_index) / (max_index - min_index)) - 1
    
    # Step 2: Calculate Demand Index
    
    # Normalize rig production as a share of global production
    df['Normalized_Rig_Production'] = df['Oil Production'] / df['Global Supply']
    
    # Apply a scaling function to moderate the impact of production
    df['Scaled_Rig_Production'] = np.log1p(df['Normalized_Rig_Production'])
    
    # Weight deepwater investments and production contributions
    weight_investment = 0.8
    weight_production = 0.2
    
    # Calculate weighted geometric mean for Demand Index
    df['Demand_Index'] = (df['Deepwater Investments'] ** weight_investment) * \
                         (df['Scaled_Rig_Production'] ** weight_production)
    
    # Normalize the Demand Index to scale it back to the range of Deepwater Investments
    df['Demand_Index'] *= df['Deepwater Investments'].mean() / df['Demand_Index'].mean()
    
    # Step 3: Filter dates up to July 1, 2029
    cutoff_date = pd.Timestamp('2029-07-01')
    df = df[df.index <= cutoff_date]
    
    # Select the relevant columns to match the structure, including dayrate columns
    final_df = df[['Min Lead Edge Dayrate', 'Avg Lead Edge Dayrate', 'Max Lead Edge Dayrate', 
                   'Supply Tightness Index', 'Scaled Supply Tightness Index', 'Demand_Index']]
    
    return final_df

# Process the dataset to get the structured output
processed_df = process_dataset(raw_df)

# Display the processed DataFrame to the user
import ace_tools as tools; tools.display_dataframe_to_user(name="Processed Jackup DataFrame", dataframe=processed_df)

