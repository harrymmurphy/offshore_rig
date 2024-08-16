
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load Datafile with rig you want to test
new_data = pd.read_csv(r'C:\Users\Harry Murphy\OneDrive\Desktop\your_file.csv', parse_dates=['Date'], index_col='Date')

# Lag the Demand_Index backwards by 1.5 years (18 months)
new_data['Demand_Index'] = new_data['Demand_Index'].shift(-18)  # Lagging backwards by 1.5 years (18 months)

# Define the forecast period to extend for 36 months from August 2024
forecast_length_36_months = 34
forecast_index_36_months = pd.date_range(start='2024-08-01', periods=forecast_length_36_months, freq='MS')

# Handle missing values by forward filling
new_data.fillna(method='ffill', inplace=True)

# Apply smoothing to the target variable and exogenous variables
smoothed_dayrate_extended = new_data['Average Lead Dayrate'].rolling(window=6, center=True).mean().fillna(method='ffill').fillna(method='bfill')
smoothed_exog_extended = new_data[['Supply Tightness Index', 'Demand_Index']].rolling(window=6, center=True).mean().fillna(method='ffill').fillna(method='bfill')

# Standardize the exogenous variables for regularization
scaler = StandardScaler()
smoothed_exog_standardized_extended = pd.DataFrame(scaler.fit_transform(smoothed_exog_extended), index=smoothed_exog_extended.index, columns=smoothed_exog_extended.columns)

# Ensure the historical data stops at July 2024
train_data_final = smoothed_dayrate_extended[:'2024-07-01']
train_exog_final = smoothed_exog_standardized_extended.loc[train_data_final.index]

# Fit the SARIMAX model on the historical data (up to July 2024)
model_final = SARIMAX(train_data_final, exog=train_exog_final, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_results_final = model_final.fit(disp=False)

# Generate the forecast for the future period using the 36-month forecast and lagged exogenous variables
forecast_exog_36_months = smoothed_exog_standardized_extended.loc[forecast_index_36_months]
forecast_values_36_months = sarimax_results_final.get_forecast(steps=forecast_length_36_months, exog=forecast_exog_36_months).predicted_mean

# Correct the index for the forecast values
forecast_values_36_months.index = forecast_index_36_months

# Combine the historical data with the forecast for comparison, ensuring no repetition of the historical data
full_forecast_36_months = pd.concat([train_data_final, forecast_values_36_months])

# Plot the original data, the forecast, and the exogenous variables
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
# Increase the padding and set the edge color to make the legend more distinct
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.5)

plt.title('SARIMAX Forecast of Average Lead Dayrate with 1.5 Year Lagged Demand Index (2024-2027)')
plt.grid(True)
plt.show()

# Extract the final forecast value for the last month in the 36-month forecast period
final_forecast_value_36_months = forecast_values_36_months.iloc[-1]
print("Final forecast value for the last month in the 36-month forecast period:", final_forecast_value_36_months)
