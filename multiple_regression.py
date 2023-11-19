#lasso
# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Reading the data from the excel file
data = pd.read_excel(r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\LCZ6\100m x 100m\Lasso\Lass.xlsx')

# Extracting independent and dependent variables
X = data[['W_area', 'AVG_Pl_DT', 'AVG_SVF', 'LambdaP']]
y = data['TincO']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Performing Lasso regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)

# Calculating the p values using statsmodels
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print('p values:', est2.pvalues)

# Calculating the r2 score for the model
y_pred = lasso_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('r2 score:', r2)

# Printing the coefficients, intercept, p values, and r2 score
print('Coefficients:', lasso_reg.coef_)
print('Intercept:', lasso_reg.intercept_)

#least square selection method
import pandas as pd
from sklearn.linear_model import Lars
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Load the dataset from the Excel file
data = pd.read_excel("C:/Users/kwame/OneDrive - The University of Western Ontario/Desktop/GOOGLE DRIVE/LCZ6/100m x 100m/Lasso/Lass.xlsx")

# Separate the independent variables from the dependent variable
X = data[["W_area", "AVG_Pl_DT", "AVG_SVF", "LambdaP"]]
y = data["TincO"]

# Create the LARS model
lars = Lars(n_nonzero_coefs=2)

# Fit the model to the data
lars.fit(X, y)

# Print the coefficients and intercept
print("Coefficients:", lars.coef_)
print("Intercept:", lars.intercept_)

# Print the r2 and p values for each independent variable
for i, col in enumerate(X.columns):
    r2 = r2_score(y, lars.predict(X))
    p = pearsonr(X[col], y)[0]
    print(f"{col} - r2: {r2:.4f}, p-value: {p:.4f}")
print('r2 values is for the entire model')



