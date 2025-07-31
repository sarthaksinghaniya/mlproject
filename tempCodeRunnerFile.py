from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("\nModel ready!")
print("X_train shape:", X_train.shape)
print("y_test shape:", y_test.shape)