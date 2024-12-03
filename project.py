import pandas as pd
import numpy as np

data=pd.read_csv("Housing.csv") #loading the dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

X = data[['bedrooms']]
y = data[['price']]

X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)

# scaler=StandardScaler()
# X_train=scaler.fit_transform(X_train)
# X_test=scaler.transform(X_test)

#model selection
model=LinearRegression()
model.fit(X_train, y_train)

#making predictions

y_predicted=model.predict(X_test)

#evaluate the model

meanabserror=mean_absolute_error(y_test,y_predicted)
meansqarederror=mean_squared_error(y_test,y_predicted)
r2 = r2_score(y_test, y_predicted)
print(data.describe())
print(f'Mean absolute error is {meanabserror}')
print(f'Mean Squared error is {meansqarederror}')
print(f'R-squared error is {r2}')

import matplotlib.pyplot as plt

plt.scatter(y_test, y_predicted,color='blue',edgecolor='black')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

bedroom_counts = data['bedrooms'].value_counts()

plt.pie(bedroom_counts, labels=bedroom_counts.index, autopct='%1.1f%%', startangle=140, colors=['blue', 'green', 'red', 'yellow'])
plt.title('Distribution of Houses by Number of Bedrooms')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(data['price'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

avg_price_by_bedrooms = data.groupby('bedrooms')['price'].mean()
avg_price_by_bedrooms.plot(kind='bar', color='lightblue', edgecolor='black')

plt.title('Average House Price by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()