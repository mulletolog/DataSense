import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Set the working directory
os.chdir('C:/Users/LimboLEG/Documents/GitHub/DataSense')

# Define the country codes
country_codes = ['CA', 'MX', 'CN', 'JP', 'DE']

# Initialize an empty DataFrame for combined data
combined_data = pd.DataFrame()

# Define the cutoff date
cutoff_date = pd.to_datetime('2023-07-01')

# Prepare the plot
plt.figure(figsize=(10, 6))

# Loop through each country's data
for code in country_codes:
    # Load reserves, exports, and exchange rate data
    res_df = pd.read_csv(f'{code}RES.csv', parse_dates=['DATE']).set_index('DATE')
    ex_df = pd.read_csv(f'{code}EX.csv', parse_dates=['DATE']).set_index('DATE')
    exr_df = pd.read_csv(f'{code}EXR.csv', parse_dates=['DATE']).set_index('DATE')

    # Apply percentage change and filter based on the cutoff date
    delta_res_df = res_df.pct_change()[res_df.index <= cutoff_date]
    ex_df['lag_export_growth'] = ex_df[f'{code}EX'].pct_change().shift(1)[ex_df.index <= cutoff_date]
    exr_df['ln_volatility'] = np.log(exr_df[f'{code}EXR'].rolling(window=12).std().shift(1) * 100)[exr_df.index <= cutoff_date]

    # Prepare the combined dataset for the regression
    country_data = pd.concat([delta_res_df, ex_df[['lag_export_growth']], exr_df[['ln_volatility']]], axis=1).dropna()

    # Perform the regression
    X = sm.add_constant(country_data[['lag_export_growth', 'ln_volatility']])
    Y = country_data[f'{code}RES']
    model = sm.OLS(Y, X).fit()
    print(f"Regression Results for {code}:")
    print(model.summary())
    print("\n")

    # Plot detrended reserves
    plt.plot(delta_res_df.index, delta_res_df[f'{code}RES'], label=code)

# Configure and display the plot
plt.xlabel('Date')
plt.ylabel('Detrended Reserves')
plt.title('Detrended International Reserves Over Time by Country')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Save the combined data to CSV
combined_data.to_csv('detrended_combined_country_data.csv')



