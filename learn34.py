import numpy as np
import pandas as pd
import os
from pathlib import Path

def create_sample_csv(filename):
    """Generates a sample 50_Startups.csv if it doesn't exist."""
    data = {
        'R&D Spend': [165349, 162597, 153441, 144372, 142107],
        'Administration': [136897, 151377, 101145, 118671, 91391],
        'Marketing Spend': [471784, 443898, 407934, 383199, 366168],
        'State': ['New York', 'California', 'Florida', 'New York', 'Florida'],
        'Profit': [192261, 191792, 191050, 182902, 166187]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created a sample dataset at: {filename}")

def find_or_create_csv(filename):
    # 1. Check if file exists in current directory
    if os.path.exists(filename):
        return filename
    
    # 2. Search in common directories
    common_dirs = [
        Path.home() / "Desktop",
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path.home() / "OneDrive/Desktop",
    ]
    
    for dir_path in common_dirs:
        file_path = dir_path / filename
        if os.path.exists(file_path):
            return str(file_path)
    
    # 3. If not found, ask the user
    print(f"\n'{filename}' not found.")
    choice = input("Do you want to create a sample file? (Y/N): ").strip().lower()
    
    if choice == 'y':
        create_sample_csv(filename)
        return filename
    else:
        return None

# --- Main Program ---
csv_file = find_or_create_csv('50_Startups.csv')

if csv_file is None:
    print("Exiting. No file available.")
    exit()

# Load data
dataset = pd.read_csv(csv_file)
print(f"Data loaded from: {csv_file}")

# Rest of your ML code...
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handle categorical 'State' column
if 'State' in dataset.columns:
    states = pd.get_dummies(dataset['State'], drop_first=True)
    x = dataset.drop('State', axis=1)
    x = pd.concat([x, states], axis=1)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nModel ready!")
print("X_train shape:", X_train.shape)
print("y_test shape:", y_test.shape)