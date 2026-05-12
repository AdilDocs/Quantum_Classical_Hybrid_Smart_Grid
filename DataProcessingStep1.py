import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "time_series_dataset.csv"
df = pd.read_csv(file_path, parse_dates=['date_time'])

df['Load (MW)'] = df[[col for col in df.columns if col.startswith("active_power_node_")]].sum(axis=1)
df['Solar Generation (MW)'] = df[[col for col in df.columns if col.startswith("renewable_active_power_node_")]].sum(axis=1)
df['Price ($/MWh)'] = df['price']

df_clean = df[['date_time', 'Load (MW)', 'Solar Generation (MW)', 'Price ($/MWh)']].copy()
df_clean.rename(columns={'date_time': 'Time'}, inplace=True)

total_scenarios = len(df_clean)

# 3. Integrate Infrastructure States (s1, s2, s3)
df_clean['s1_Gen'] = 1
df_clean['s2_Line'] = 1
df_clean['s3_XFMR'] = 1

base_gen_capacity = 3040
df_clean['P_available'] = df_clean['Solar Generation (MW)'] + base_gen_capacity
df_clean['Label_y'] = (df_clean['P_available'] >= df_clean['Load (MW)']).astype(int)

counts = df_clean['Label_y'].value_counts()
extreme_total = counts[0]
reliable_total = counts[1]

print("-" * 30)
print("DATASET STATISTICS SUMMARY")
print("-" * 30)
print(f"Total Scenarios: {total_scenarios:,}")
print(f"Extreme (0):    {extreme_total:,}")
print(f"Reliable (1):   {reliable_total:,}")
print("\nPROPOSED STRATIFIED SPLIT (80/20):")
print(f"Training Set (80%):")
print(f"  - Extreme:  {extreme_total} * 0.8 = {int(extreme_total * 0.8):,}")
print(f"  - Reliable: {reliable_total} * 0.8 = {int(reliable_total * 0.8):,}")
print(f"Testing Set (20%):")
print(f"  - Extreme:  {extreme_total} * 0.2 = {int(extreme_total * 0.2):,}")
print(f"  - Reliable: {reliable_total} * 0.2 = {int(reliable_total * 0.2):,}")
print("-" * 30)

df_clean.to_csv("time_series_clean.csv", index=False)

# 6. Visualization
plt.figure(figsize=(12, 10))
colors = ['#1f77b4', '#2ca02c', '#d62728']
titles = ['Total System Load Profile (MW)', 'Renewable Generation Profile (MW)', 'Market Price Profile ($/MWh)']
cols = ['Load (MW)', 'Solar Generation (MW)', 'Price ($/MWh)']

for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(df_clean['Time'], df_clean[cols[i]], color=colors[i], linewidth=0.8)
    plt.ylabel(cols[i])
    plt.title(titles[i])
    plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()