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

**Output**
<img width="772" height="294" alt="image" src="https://github.com/user-attachments/assets/90d8b5fc-354d-4272-b113-7926712b9d99" />

<img width="893" height="522" alt="image" src="https://github.com/user-attachments/assets/15b2ef78-24b0-4102-ab87-13b329576483" />

<img width="1132" height="509" alt="image" src="https://github.com/user-attachments/assets/e038cbbb-0551-4c7e-8ca3-e4a21158c925" />

<img width="1073" height="530" alt="image" src="https://github.com/user-attachments/assets/74807da0-6e1b-4a13-9af8-0d899cdff03e" />


**Result**
The bivariate analysis shows a strong positive correlation between daily COVID-19 cases and deaths.
The linear regression model successfully predicts deaths from case counts with good accuracy (R² ≈ 0.8).
This experiment demonstrates how linear models can be used for simple real-time forecasting in public health analytics.
Display the predicted future deaths along with correlation, R², and RMSE values.
