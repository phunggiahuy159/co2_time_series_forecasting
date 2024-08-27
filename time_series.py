import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('co2_weekly_mlo.csv')


df['time']=df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
df['time']=pd.to_datetime(df['time'])
df.drop(['year', 'month', 'day'], axis=1, inplace=True)


df['average'].replace(-999.99, np.nan, inplace=True)


df['average'].interpolate(method='linear', inplace=True)

time=df['time']
'''some visualization'''
# fig,ax=plt.subplots()
# ax.plot(df['time'],df['average'])
# ax.set_xlabel('time')
# ax.set_ylabel('CO2')
# plt.show()


def create_recursive(df, window_size):
    for i in range(1, window_size + 1):
        df[f'co2_{i}'] = df['average'].shift(-i)
    df['result'] = df['average'].shift(-window_size-1)
    res = df.dropna(axis=0)
    return res

window=5

new_data=create_recursive(df, window)

# Define features (X) and target (y)
X=new_data.drop(['average', 'time', 'result'], axis=1)
y=new_data['result']

# Split the data into training and testing sets
num_samples=len(y)
train_size=0.8
cut=int(train_size * num_samples)

X_train=X[:cut]
y_train=y[:cut]
X_test=X[cut:]
y_test=y[cut:]

# Fit the model
model=LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred=model.predict(X_test)

# Calculate and print the R^2 score
print("R^2 Score:", r2_score(y_test, y_pred))

fig,ax=plt.subplots()
ax.plot(time[cut:cut+len(y_pred)], y_pred, label='Predicted', color='green')
ax.plot(time[cut:cut+len(y_test)], y_test, label='Actual', color='red')
ax.legend(['Predicted','Actual'])
plt.show()
