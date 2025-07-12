import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate mock weather data for 30 days
np.random.seed(42)  # For consistent results

days = pd.date_range(start='2025-07-01', periods=30)
temperature = np.random.normal(loc=32, scale=3, size=30)       # Avg temp 32°C
humidity = np.random.normal(loc=70, scale=10, size=30)          # Avg humidity
rainfall = np.random.exponential(scale=5, size=30)              # Random daily rain

# 2. Create DataFrame
df = pd.DataFrame({
    'Date': days,
    'Temperature (°C)': temperature,
    'Humidity (%)': humidity,
    'Rainfall (mm)': rainfall
})

# 3. Display basic stats
print("📊 Basic Summary:")
print(df.describe())

# 4. Plot temperature trend
plt.figure(figsize=(10, 4))
plt.plot(df['Date'], df['Temperature (°C)'], marker='o', color='tomato')
plt.title("🌡️ Temperature Trend (July 2025)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Plot humidity & rainfall on same chart
fig, ax1 = plt.subplots(figsize=(10, 4))

ax1.set_xlabel('Date')
ax1.set_ylabel('Humidity (%)', color='blue')
ax1.plot(df['Date'], df['Humidity (%)'], color='blue', marker='x', label='Humidity')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Rainfall (mm)', color='green')
ax2.plot(df['Date'], df['Rainfall (mm)'], color='green', marker='s', label='Rainfall')
ax2.tick_params(axis='y', labelcolor='green')

plt.title("💧 Humidity vs Rainfall")
plt.tight_layout()
plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop(columns='Date').corr(), annot=True, cmap='coolwarm')
plt.title("📈 Correlation Heatmap")
plt.tight_layout()
plt.show()

# 7. Smart insights
hottest_day = df.loc[df['Temperature (°C)'].idxmax()]
print(f"\n🔥 Hottest day: {hottest_day['Date'].date()} ({hottest_day['Temperature (°C)']:.1f}°C)")

rainiest_day = df.loc[df['Rainfall (mm)'].idxmax()]
print(f"🌧️ Most rainy day: {rainiest_day['Date'].date()} ({rainiest_day['Rainfall (mm)']:.1f} mm)")
