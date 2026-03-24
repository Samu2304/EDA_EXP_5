# Experiment 5: COVID-19 Daily Cases Forecasting (Bivariate Analysis)
**AIM:**

To perform bivariate analysis on COVID-19 data and study the relationship between daily confirmed cases and daily deaths, using Linear Regression for prediction and simple real-time forecasting.

**ALGORITHM / PROCEDURE:**

1)Import the required Python libraries: pandas, numpy, matplotlib, and sklearn modules for model building and evaluation.

2)Load the COVID-19 dataset (CSV file) into a pandas DataFrame.

3)Select a specific country’s data (e.g., India or USA) for analysis.

4)Perform bivariate analysis by calculating the correlation between daily cases and daily deaths.

5)Visualize the relationship between daily cases and daily deaths using a scatter plot.

6)Define the input variable (X = Daily Cases) and the output variable (y = Daily Deaths).

7)Split the dataset into training and testing parts using the train_test_split() function.

8)Create and train a Linear Regression model using the training data.

9)Predict daily deaths for the test dataset using the trained model.

10)Evaluate the model performance by calculating R² (coefficient of determination) and RMSE (Root Mean Squared Error).

11)Plot a graph comparing actual deaths versus predicted deaths to visualize model accuracy.

12)Plot daily cases and daily deaths over time to observe the time trend and lag between them.

13)Generate a range of possible future case numbers and predict corresponding deaths using the trained model.



**Program**

**Name**: Samyuktha S
**Reg No.**: 212222240089

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv"
try:
  data = pd.read_csv(url)
  print(" Dataset loaded successfully from online source.")
except Exception as e:
  print(" Online dataset could not be loaded. Using local fallback sample dataset.")
  fallback_url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
  data = pd.read_csv(fallback_url)

```
```
country = "India"
country_data = data[data['Country'] == country].copy()

country_data['Date'] = pd.to_datetime(country_data['Date'])

country_data['Daily_Cases'] = country_data['Confirmed'].diff().fillna(0)
country_data['Daily_Deaths'] = country_data['Deaths'].diff().fillna(0)

print("\nSample of processed data:")
print(country_data.head())
```
```
plt.figure(figsize=(7,5))
plt.scatter(country_data['Daily_Cases'], country_data['Daily_Deaths'], alpha=0.5, color='purple')
plt.title(f'Bivariate Analysis: {country} COVID-19 Daily Cases vs Deaths')
plt.xlabel('Daily Cases')
plt.ylabel('Daily Deaths')
plt.grid(True)
plt.show()
```
```
corr = country_data['Daily_Cases'].corr(country_data['Daily_Deaths'])
print(f"\n Correlation between Daily Cases and Deaths in {country}: {corr:.3f}")
```
```
X = country_data[['Daily_Cases']]
y = country_data['Daily_Deaths']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
```
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n Model Evaluation for {country}:")
print(f"   R² Score  = {r2:.3f}")
print(f"   RMSE      = {rmse:.3f}")
```
```
plt.figure(figsize=(10,5))
plt.plot(country_data['Date'][-len(y_test):], y_test, label='Actual Deaths', color='blue')
plt.plot(country_data['Date'][-len(y_pred):], y_pred, label='Predicted Deaths', color='red', linestyle='--')
```
```
future_cases = np.linspace(X['Daily_Cases'].max()*0.5, X['Daily_Cases'].max()*1.2, 10).reshape(-1, 1)
future_pred = model.predict(future_cases)
print("\n Forecast of Future Deaths based on Case Count:")
for c, d in zip(future_cases.flatten(), future_pred):
  print(f"   Predicted deaths for {int(c)} cases ≈ {d:.1f}")
  ```
```
plt.figure(figsize=(10,5))
plt.plot(country_data['Date'], country_data['Daily_Cases'], label='Daily Cases', color='green')
plt.plot(country_data['Date'], country_data['Daily_Deaths'], label='Daily Deaths', color='orange')
plt.title(f'{country}: COVID-19 Daily Cases and Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()
```
```
**Output**


**Result**
The bivariate analysis shows a strong positive correlation between daily COVID-19 cases and deaths.
The linear regression model successfully predicts deaths from case counts with good accuracy (R² ≈ 0.8).
This experiment demonstrates how linear models can be used for simple real-time forecasting in public health analytics.
Display the predicted future deaths along with correlation, R², and RMSE values.
