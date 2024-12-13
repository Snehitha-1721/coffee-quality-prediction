
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


file_path = '/content/df_arabica_clean.csv'
df = pd.read_csv(file_path)


features = ['Moisture Percentage', 'Category One Defects', 'Altitude', 'Total Cup Points']
df = df[features]


df = df.dropna()
def convert_altitude(altitude):
    try:
        return float(altitude)  
    except ValueError:
        if '-' in altitude:  
            lower, upper = map(float, altitude.split('-'))
            return (lower + upper) / 2
        else:
            return np.nan  

df['Altitude'] = df['Altitude'].apply(convert_altitude)
df = df.dropna()  # Drop rows with NaN values after conversion

# Define features (X) and target (y)
X = df.drop("Total Cup Points", axis=1)  # Features
y = df["Total Cup Points"]               # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Total Cup Points')
plt.ylabel('Predicted Total Cup Points')
plt.title('Actual vs Predicted Total Cup Points')
plt.show()

# Optional: Feature importance (coefficients of the linear model)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Importance:")
print(feature_importance)
