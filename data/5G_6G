import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the provided dataset
file_path = r'C:\Users\Harry Murphy\OneDrive\Desktop\5G_6G_adjusted.csv'
new_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Function to process the dataset
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
    
    # Return the processed DataFrame
    return df

# Process the dataset
processed_data = process_dataset(new_data)

# Lag the Demand_Index backwards by 1.5 years (18 months)
processed_data['Demand_Index'] = processed_data['Demand_Index'].shift(-18)

# Define the forecast period to end on January 1, 2027
forecast_index_36_months = pd.date_range(start='2024-08-01', end='2027-01-01', freq='MS')
forecast_length_36_months = len(forecast_index_36_months)

# Handle missing values by forward filling
processed_data.fillna(method='ffill', inplace=True)

# Apply smoothing to the target variable and exogenous variables
smoothed_dayrate_extended = processed_data['Avg Lead Dayrate'].rolling(window=6, center=True).mean().fillna(method='ffill').fillna(method='bfill')
smoothed_exog_extended = processed_data[['Supply Tightness Index', 'Demand_Index']].rolling(window=6, center=True).mean().fillna(method='ffill').fillna(method='bfill')

# Standardize the exogenous variables for regularization
scaler = StandardScaler()
smoothed_exog_standardized_extended = pd.DataFrame(scaler.fit_transform(smoothed_exog_extended), index=smoothed_exog_extended.index, columns=smoothed_exog_extended.columns)

# Ensure the historical data stops at July 2024
train_data_final = smoothed_dayrate_extended[:'2024-07-01']
train_exog_final = smoothed_exog_standardized_extended.loc[train_data_final.index]

# Fit the SARIMAX model on the historical data (up to July 2024)
model_final = SARIMAX(train_data_final, exog=train_exog_final, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_results_final = model_final.fit(disp=False)

# Generate the forecast for the future period using the forecast period and lagged exogenous variables
forecast_exog_36_months = smoothed_exog_standardized_extended.loc[forecast_index_36_months]
forecast_values_36_months = sarimax_results_final.get_forecast(steps=forecast_length_36_months, exog=forecast_exog_36_months).predicted_mean

# Correct the index for the forecast values
forecast_values_36_months.index = forecast_index_36_months

# Combine the historical data with the forecast for comparison, ensuring no repetition of the historical data
full_forecast_36_months = pd.concat([train_data_final, forecast_values_36_months])

# Plot the original data, the forecast, and the exogenous variables up to January 1, 2027
fig, ax1 = plt.subplots(figsize=(14, 8))

# Plot historical dayrate data on the primary y-axis
ax1.plot(train_data_final.index, train_data_final, label='Historical Average Lead Dayrate (Smoothed)', color='blue', linestyle='-')
ax1.plot(forecast_values_36_months.index, forecast_values_36_months, label='Forecasted Average Lead Dayrate (SARIMAX, Regularized)', color='red', linestyle='--')
ax1.set_xlabel('Date')
ax1.set_ylabel('Average Lead Dayrate', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Restrict the exogenous variables to end on January 1, 2027
x_axis_limit = pd.to_datetime('2027-01-01')

# Create a second y-axis for the Supply Tightness Index
ax2 = ax1.twinx()
ax2.plot(smoothed_exog_standardized_extended.loc[:x_axis_limit].index, smoothed_exog_standardized_extended.loc[:x_axis_limit]['Supply Tightness Index'], label='Supply Tightness Index', color='green')
ax2.set_ylabel('Supply Tightness Index', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Create a third y-axis for the lagged Demand Index
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 34))  # Move the third y-axis out
ax3.plot(smoothed_exog_standardized_extended.loc[:x_axis_limit].index, smoothed_exog_standardized_extended.loc[:x_axis_limit]['Demand_Index'], label='Demand Index (Lagged)', color='orange')
ax3.set_ylabel('Demand Index', color='orange')
ax3.tick_params(axis='y', labelcolor='orange')

# Add vertical line to indicate where the forecast starts
ax1.axvline(x=pd.to_datetime('2024-07-01'), color='black', linestyle='--', label='Forecast Start')

# Combine legends from all axes with a dense background
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()

# Create the legend with a dense background (opaque)
legend = ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left', framealpha=1, facecolor='white')
legend.get_frame().set_linewidth(1.5)

plt.title('SARIMAX Forecast of Average Lead Dayrate with 1.5 Year Lagged Demand Index (2024-2027)')
plt.grid(True)
plt.show()

# Extract the final forecast value for the last month in the forecast period (January 2027)
final_forecast_value_36_months = forecast_values_36_months.iloc[-2]
print("Final forecast value for January 2027:", final_forecast_value_36_months)


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define a validation period (e.g., the last 12 months of historical data)
validation_period = '2020-08-01'  # Adjust this period as needed
validation_data = smoothed_dayrate_extended[validation_period:'2024-07-01']
validation_exog = smoothed_exog_standardized_extended.loc[validation_data.index]

# Generate predictions for the validation period
validation_predictions = sarimax_results_final.get_prediction(start=validation_period, end='2024-07-01', exog=validation_exog).predicted_mean

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(validation_data, validation_predictions))
mae = mean_absolute_error(validation_data, validation_predictions)

print("RMSE for the validation period:", rmse)
print("MAE for the validation period:", mae)

# Plot the validation results
plt.figure(figsize=(10, 6))
plt.plot(validation_data.index, validation_data, label='Actual Average Lead Dayrate (Validation Period)', color='blue')
plt.plot(validation_predictions.index, validation_predictions, label='Predicted Average Lead Dayrate (Validation Period)', color='red', linestyle='--')
plt.title('Validation Period: Actual vs Predicted Dayrates')
plt.xlabel('Date')
plt.ylabel('Average Lead Dayrate')
plt.legend()
plt.show()
file_1 = r'C:\Users\Harry Murphy\OneDrive\Desktop\5G_6G_forecast.csv'
full_forecast_36_months.to_csv(file_1)

