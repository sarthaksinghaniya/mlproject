import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set(style="whitegrid")
np.random.seed(42)

# Generate synthetic weather data
days = pd.date_range(start='2025-07-01', periods=30)
temperature = np.random.normal(loc=32, scale=3, size=30)
humidity = np.random.normal(loc=70, scale=10, size=30)
rainfall = np.random.exponential(scale=5, size=30)

# Create DataFrame
df = pd.DataFrame({
    'Date': days,
    'Temperature (°C)': temperature,
    'Humidity (%)': humidity,
    'Rainfall (mm)': rainfall
})

# --- Basic Summary ---
print("📊 Basic Weather Summary for July 2025\n")
print(df.describe())

# --- Temperature Trend ---
plt.figure(figsize=(10, 4))
plt.plot(df['Date'], df['Temperature (°C)'], marker='o', color='tomato')
plt.title("🌡️ Temperature Trend (July 2025)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Humidity vs Rainfall ---
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.set_xlabel('Date')
ax1.set_ylabel('Humidity (%)', color='blue')
ax1.plot(df['Date'], df['Humidity (%)'], color='blue', marker='x')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Rainfall (mm)', color='green')
ax2.plot(df['Date'], df['Rainfall (mm)'], color='green', marker='s')
ax2.tick_params(axis='y', labelcolor='green')

plt.title("💧 Humidity vs Rainfall - July 2025")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop(columns='Date').corr(), annot=True, cmap='coolwarm')
plt.title("📈 Correlation Between Weather Metrics")
plt.tight_layout()
plt.show()

# --- Smart Insights ---
hottest = df.loc[df['Temperature (°C)'].idxmax()]
rainiest = df.loc[df['Rainfall (mm)'].idxmax()]

print(f"\n🔥 Hottest Day: {hottest['Date'].date()} → {hottest['Temperature (°C)']:.1f}°C")
print(f"🌧️ Rainiest Day: {rainiest['Date'].date()} → {rainiest['Rainfall (mm)']:.1f} mm")
