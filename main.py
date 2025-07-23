import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings("ignore") #suppress warning messages


# Choose a threshold 
threshold = -0.5 
generate_plot = False
debugging = False

# Coins list
coins = ['btc','eth','xrp', 'usdt', 'sol', 'bnb', 'doge', 'STETH', 'ada', 'trx', 'AVAX', 'LINK', 'WSTETH', 'shib', 'ton', 'WBTC', 'sui', 'xlm', 'bch', 'pepe'] #'usdt' #'usdc'   #'btc','eth','xrp', 'sol', 'bnb', 'doge', 'STETH', 'ada', 'trx', 'AVAX', 'LINK', 'WSTETH', 'shib', 'ton', 'WBTC', 'sui', 'xlm', 'dot', 'bch', 'pepe'

# API configuration
API_KEY = "" #YOUR API KEY HERE
BASE_URL = "https://data.messari.io/api/v1"





# Function to get the data
def fetch_data(asset, start_date, end_date):
    endpoint = f"/assets/{asset}/metrics/price/time-series"
    params = {
        "start": start_date,
        "end": end_date,
        "interval": "1d",
    }
    headers = {"x-messari-api-key": API_KEY}
    response = requests.get(BASE_URL + endpoint, headers=headers, params=params)

    data = response.json()
    time_series = data['data']['values']
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(time_series, columns=columns)

    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    return df



# Function to calculate ATH and Drawdown 
def ath_with_drawdown(df):

    '''This function does this:

    - Create a column in the df called ATH Value where it stores the highest 'High' value until that date

    - Create a column 'Drawdown' which is calculated as a percentage decrease from the ath 

    - Create a column which contains days since the last ath

    - Selects the deepest drawdown (maximum negative drawdown) that occurs after each unique value of the ATH

    - Filter the dataframe to keep only the relevant column and the relevant dates (the days where the deepest drawdown was recorded)
    '''

    df['ATH Value'] = df['High'].cummax() # create column ath
    df['Drawdown'] = ((df['Low'] - df['ATH Value']) / df['ATH Value'])  # calculate drawdown


    # Calculate days since the last ath 
    #--- begin code from ChatGPT (GitHub Copilot)---
    last_ath_indices = df['ATH Value'] == df['High']
    df['Days Since ATH'] = df.index.to_series().where(last_ath_indices).ffill().fillna(df.index[0]).astype(int) 
    df['Days Since ATH'] = df.index - df['Days Since ATH'] # Calculate days since the last ATH
    #--- end code from ChatGPT (GitHub Copilot)---


    # keep only the minimum value of drawdown corresponding to each ath
    #--- begin code from ChatGPT (GitHub Copilot)---
    min_drawdown_indices = df.groupby('ATH Value')['Drawdown'].idxmin()
    filtered_df = df.loc[min_drawdown_indices]
    #--- end code from ChatGPT (GitHub Copilot)---

    #print(filtered_df)   # Useful for Debug

    # keep only relevant columns
    filtered_df = filtered_df[['Date', 'ATH Value', 'Drawdown', 'Days Since ATH']].reset_index(drop=True)
    return filtered_df



# Calculate proportion of drawdown under the threshold
def calculate_drawdown_statistics(filtered_df, exclude_recent_days, threshold, debug = False):
    """
    Calculate proportion of drawdown under the threshold, as (total number of drawdown)/(number of drawdown under the threshold) 

    This function can exclude certain days from the calculation (will be used when 2 ath's are close to each other and the drawdown value would result in a small number)

    Calculate statistics for the drawdowns
    """
    #Filter data based on the excluded days
    filtered_data = filtered_df[filtered_df['Days Since ATH'] >= exclude_recent_days]

    if debug:
        print(f"\nFiltered data after excluding {exclude_recent_days} days:")
        print(filtered_data)

    n_drawdowns = len(filtered_data)  # total number of drawdowns

    # Count drawdown below the threshold
    n_below_threshold = (filtered_data['Drawdown'] <= threshold).sum()

    # Calclulate proportion under threshold
    proportion = n_below_threshold / n_drawdowns

    # Calculate statistics of drawdowns
    mean_drawdown = filtered_data['Drawdown'].mean()
    std_drawdown = filtered_data['Drawdown'].std()


    return n_drawdowns, n_below_threshold, proportion, mean_drawdown, std_drawdown




# Find the threshold for which 100% of the drawdowns are below it
def find_drawdown_for_full_proportion(filtered_df, exclude_recent_days):
    """
    Finds the deepest drawdown value such that 100% of the drawdowns 
    are less than or equal to it, excluding drawdowns that occurred within a 
    specified number of days from the last ATH (exclude_recent_days)
    
    """
    # Filter drawdown close to the ath
    filtered_data = filtered_df[filtered_df['Days Since ATH'] >= exclude_recent_days]

    max_drawdown_full_proportion = filtered_data['Drawdown'].max()
    
    # find the highest drawdown
    return max_drawdown_full_proportion




# Function to generate plots
def generate_plots(df, coin, filtered_df, days_list, proportions, thresholds, show_plots=True):

    '''Function to generate plots:
    - Time series Plot
    - Proportion of Drawdown under the threshold plot
    - Threshold drawdown Values for 100% Proportion plot

    Used ChatGpt to refine the plots'''

    if show_plots:

        # Time series plot
        plt.figure(figsize=(12, 6))

        # Plot the high prices
        plt.plot(df['Date'], df['High'], label='Price (High)', color='black', alpha=1, linewidth=1.5) 

        # Plot the ATH points
        ath_indices = df[df['High'] == df['ATH Value']]
        plt.scatter(ath_indices['Date'], ath_indices['ATH Value'], color='red', label='ATH (All-Time High)', s=10, zorder=5)  

        # Plot the min drawdowns
        min_drawdown_indices = df.loc[df.groupby('ATH Value')['Low'].idxmin()]  # Get the min drawdown points for each ATH
        plt.scatter(min_drawdown_indices['Date'], min_drawdown_indices['Low'], color='green', label='Min Drawdown', s=10, zorder=5) 

        # Add labels and legend
        plt.title(f"Time Series of {coin.upper()}", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()




        # Proportion under the threshold plot
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("Blues", len(days_list))
        plt.bar([f"{days} days" for days in days_list], proportions, color=colors, alpha=0.8, edgecolor='black')

        # Add bar height annotations
        for i, proportion in enumerate(proportions):
            plt.text(i, proportions[i] + 0.01, f"{proportion:.2%}", ha='center', va='bottom', fontsize=10, color='black')

        plt.title(f"Proportion of Drawdowns Below the Threshold - {coin.upper()}", fontsize=16, fontweight='bold', color='navy')
        plt.xlabel("Days Excluded After ATH", fontsize=12, fontweight='bold')
        plt.ylabel("Proportion (%)", fontsize=12, fontweight='bold')
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()




        # Plot for drawdown Values for 100% Proportion
        plt.figure(figsize=(10, 6))

        # Create bar plot 
        plt.bar([f"{days} days" for days in days_list],  [thresholds[days] for days in days_list],  color=sns.color_palette("Purples", len(days_list)), alpha=0.8, edgecolor='black', linewidth=0.8)

        # Title and axis labels
        plt.title(f"Drawdown Thresholds for 100% Proportion - {coin.upper()}", fontsize=16, fontweight='bold', color='darkred')
        plt.xlabel("Days Excluded After ATH", fontsize=12, fontweight='bold')
        plt.ylabel("Drawdown Threshold (%)", fontsize=12, fontweight='bold')

        # Add values on top of the bars
        for i, threshold in enumerate([thresholds[days] for days in days_list]):
            plt.text(i, threshold + 0.02, f"{threshold:.2%}", ha='center', va='bottom', fontsize=10, color='black')

        # Improved grid style
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

        # Add legend
        plt.legend(["Threshold for 100% Proportion"], loc='lower left', fontsize=10)

        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()








# Main cicle for the analysis

results_df = pd.DataFrame()  # DataFrame to collect all the data at the end

for coin in coins:
    # Fetch Data (we have to use two API calls because a single call cannot retrieve all the data since the coin inception)
    df_2010_2016 = fetch_data(coin, "2009-01-01", "2018-12-31")
    df_2017_2023 = fetch_data(coin, "2019-01-01", "2023-12-31")

    # Merge the dataframes
    df = pd.concat([df_2010_2016, df_2017_2023], ignore_index=True)

    # Calculate ATH and drawdowns
    filtered_df = ath_with_drawdown(df)

    # Print the number of drawdowns
    total_drawdowns = len(filtered_df)
    print("-" * 50)
    print(f"--- Analysis of: {coin.upper()} ---")
    print(f"Total number of ATH: {total_drawdowns}")
    print("-" * 50)



    # --- USED THE HELP OF CHATGPT ---

    # Calculate the proportion for the selected data
    days_list = [1, 7, 14, 30]  # Days to exclude after ATH
    proportions = []  # List to store proportions of drawdowns below the threshold
    thresholds = {}  # Dictionary to map days to their thresholds

    for days in days_list:
        n_drawdowns, n_below_threshold, proportion, mean_drawdown, std_drawdown = calculate_drawdown_statistics(filtered_df, exclude_recent_days=days, threshold=threshold, debug=debugging)

        # Append the proportion of drawdowns below the threshold for this exclusion period
        proportions.append(proportion)

        # Find the threshold where 100% of the drawdowns are below it
        threshold_value = find_drawdown_for_full_proportion(filtered_df, exclude_recent_days=days)
        thresholds[days] = threshold_value  # Map the exclusion period to the threshold

        # Print the results for the current exclusion period
        print(f"Excluding {days} days after ATH:")
        print(f"- Number of drawdowns: {n_drawdowns}")
        print(f"- Drawdowns below the threshold: {n_below_threshold}")
        print(f"- Proportion: {proportion:.2%}")
        print(f"- Mean Drawdown: {mean_drawdown:.4f}")
        print(f"- Standard Deviation of Drawdown: {std_drawdown:.4f}")
        print(f"- Drawdown Value such that the hypothesis is true: {threshold_value:.4f}")
        print("-" * 50)

    # Add results to the centralized DataFrame
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                'Coin': [coin],
                'Excluded Days': [days],
                'Num Drawdowns': [n_drawdowns],
                'Drawdowns Below Threshold': [n_below_threshold],
                'Proportion': [proportion],
                'Drawdown for 100% Proportion': [threshold_value]
            })
        ], ignore_index=True)
    
    # --- END HELP OF CHATGPT ---

    print(('-')*50)

    # Generate plots for the current coin
    generate_plots(df, coin, filtered_df, days_list, proportions, thresholds, generate_plot)



#DataFrame Visualization
pd.set_option('display.max_rows', None) #display all the rows in the dataframe
print(results_df) 



print('-'*100)

# VISUALIZE DATAFRAMES FOR DIFFERENT EXLUSION PERIODS
#--- begin code from ChatGPT (GitHub Copilot)---
dfs_by_days = {days: results_df[results_df["Excluded Days"] == days] for days in results_df["Excluded Days"].unique()}
#--- end code from ChatGPT (GitHub Copilot)---

# Visualizzare i DataFrame per ogni valore di "Excluded Days"
for days, df_split in dfs_by_days.items():
    print(f"\nDataFrame for Excluded Days = {days}:\n")
    print(df_split)
    print('-'*100)








print('-'*100)

print('Expectations of drawdown:')

#--- begin code from ChatGPT (GitHub Copilot)---

# Extract the strategy based on the lowest value of "Threshold for 100% Proportion"
strategy_list = []

for coin in results_df['Coin'].unique():
    # Filter the data for each coin
    coin_data = results_df[results_df['Coin'] == coin]
    
    # Find the row with the minimum "Threshold for 100% Proportion"
    min_threshold_row = coin_data.loc[coin_data['Drawdown for 100% Proportion'].idxmin()]
    
    # Extract the relevant details
    min_threshold = min_threshold_row['Drawdown for 100% Proportion']
    wait_days = int(min_threshold_row['Excluded Days'])
    
    # Create the strategy
    strategy = (
        f"For {coin.upper()}, after an ATH occurs, we wait {wait_days} days, "
        f"and if no other ATH occurs, the coin is expected to drop at leats by {min_threshold:.2%}."
    )
    
    strategy_list.append(strategy)

# Print the strategy for each coin
for strat in strategy_list:
    print(strat)

#--- End code from ChatGPT (GitHub Copilot)---









'''We hereby certify that
 - We have written the program ourselves except for clearly marked pieces of code and mentions of Generative AI
 - We have tested the program and it ran without crashing
 
 Niccolo' Lecce
 Marco Gasparetti
 Francesco Federico
 '''
