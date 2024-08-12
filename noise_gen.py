import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data_set = [
  [60.9 ,61],[61.81, 62.1],
  [62.28, 62.45],
  [61.2, 61.75],
  [65.87, 70.85],
  [61.31, 62.5],
  [61.3, 61.55],
  [61.84, 80.2],
  [62.33, 72.1],
  [62.1, 61.9]
]
df = pd.DataFrame(data_set, columns=['Value1', 'Value2'])

def laggin(df, lags):
    
    df_lagged = df.copy()     
                           
    for lags in range(1, lags + 1):
        df_lagged[f'Value1_lag_{lags}'] = df_lagged['Value1'].shift(lags)
        
        df_lagged[f'Value2_lag_{lags}'] = df_lagged['Value2'].shift(lags)
        
    df_lagged.dropna(inplace=True)
    
    return df_lagged

lag_no = 4

lagpd = laggin(df, lag_no)

X = lagpd.drop(columns=['Value1', 'Value2'])


y = lagpd['Value1']

def noise_factor(X, y, noise_level=0.1, n_samples=10):
    
    X_noise = X.copy()
    
    y_noise = y.copy()

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_level, X.shape)
        X_noise = pd.concat([X_noise, pd.DataFrame(X + noise, columns=X.columns)], ignore_index=True)
        
        y_noise = pd.concat([y_noise, pd.Series(y)], ignore_index=True)

    return X_noise, y_noise

X_noisy, y_noisy = noise_factor(X, y, noise_level=0.5, n_samples=10)

X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_noisy, test_size=0.2, shuffle=False)

Regression_model_g = LinearRegression()

Regression_model_g.fit(X_train, y_train)

y_pred = Regression_model_g.predict(X_test)

mean_squared_e = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mean_squared_e:.2f}')


print("Predictions:", y_pred)

plt.figure(figsize=(12, 6))

plt.plot(df['Value1'], label='Original Values', marker='o', linestyle='-', color='blue')

pred_Value = np.arange(len(y_test)) + len(df) - len(y_test)


plt.plot(pred_Value, y_pred, label='predictions', marker='o', linestyle='--', color='red')

plt.xlabel('Period')
plt.ylabel('Value1')
plt.title('Forecasting with Linear Regression and Noise Augmentation')
plt.legend()
plt.grid(True)
plt.show()