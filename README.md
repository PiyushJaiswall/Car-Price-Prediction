# Car Price Prediction

## Overview
Car Price Prediction is a machine learning project that aims to predict the price of cars based on various features like model, year, transmission type, mileage, fuel type, tax, mpg, and engine size. The dataset used for this project is obtained from a CSV file.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Installation
To run this project, ensure you have Python installed along with the required libraries. You can install the dependencies using:

```bash
pip install numpy pandas matplotlib sklearn pandas-profiling
```

## Dataset
The dataset is loaded from a CSV file:

```python
import pandas as pd

df = pd.read_csv("C:/Users/KIIT/Documents/audi.csv")
display(df.head())
```

The dataset contains the following columns:
- **model**: Car model
- **year**: Manufacturing year
- **price**: Car price
- **transmission**: Transmission type
- **mileage**: Distance covered by the car
- **fuelType**: Type of fuel used
- **tax**: Road tax
- **mpg**: Miles per gallon
- **engineSize**: Engine size in liters

## Exploratory Data Analysis
EDA is performed to understand the dataset's structure and detect missing values:

```python
import pandas_profiling as pf
display(pf.ProfileReport(df))
```

Checking for missing values and data types:

```python
display(df.isna().sum())
print(df.info())
display(df.describe())
```

## Preprocessing
### Encoding Categorical Variables
Using `LabelEncoder` and `OneHotEncoder` for categorical features:

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

le1 = LabelEncoder()
df['model'] = le1.fit_transform(df['model'])
le2 = LabelEncoder()
df['fuelType'] = le2.fit_transform(df['fuelType'])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(df.iloc[:, [0,1,3,4,5,6,7,8]].values)
```

### Standardizing Data

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
```

### Splitting the Dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, df['price'].values, test_size=0.2, random_state=0)
```

## Model Training
### Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X_train, Y_train)
```

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
```

## Evaluation
Using `r2_score` and `mean_absolute_error` to evaluate models:

```python
from sklearn.metrics import r2_score, mean_absolute_error

rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print('Random Forest R2 Score:', r2_score(Y_test, rf_pred))
print('Random Forest MAE:', mean_absolute_error(Y_test, rf_pred))

print('Linear Regression R2 Score:', r2_score(Y_test, lr_pred))
print('Linear Regression MAE:', mean_absolute_error(Y_test, lr_pred))
```

## Results
- **Random Forest**: Higher accuracy with an R2 score of ~0.95 and lower MAE.
- **Linear Regression**: Lower accuracy with an R2 score of ~0.79 and higher MAE.

## Conclusion
Random Forest performs significantly better than Linear Regression for car price prediction. Future improvements may include hyperparameter tuning and experimenting with other regression models.

