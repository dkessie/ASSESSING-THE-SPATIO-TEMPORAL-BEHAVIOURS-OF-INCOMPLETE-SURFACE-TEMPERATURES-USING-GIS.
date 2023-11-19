

#daytime regression




import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Define the independent variables (x) and the dependent variable (y)
x = df[['S_HGT_AGL', 'LambdaP', 'W_area', 'AVG_Pl_DT']]
y = df['TincO']

# Add a constant column to the independent variables
x = sm.add_constant(x)

# Create the multiple regression model
model = sm.OLS(y, x)

# Fit the model to the data
results = model.fit()

# Get the predicted values
predicted_values = results.predict()

# Create a scatter plot of actual TincO (x-axis) vs. predicted TincO (y-axis)
plt.scatter(y, predicted_values, facecolors='none', edgecolors='blue')
plt.xlabel('Calculated $T_{ped}$ (°C)')
plt.ylabel('Predicted $T_{ped}$ (°C)')
plt.title('Regression Model: Calculated vs. Predicted $T_{ped}$ (°C)')

# Get the minimum and maximum values for x and y
x_min = y.min()
x_max = y.max()
y_min = predicted_values.min()
y_max = predicted_values.max()

# Plot the identity line
min_val = min(x_min, y_min)
max_val = max(x_max, y_max)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values
plt.xlim(min_val - 1, max_val + 1)
plt.ylim(min_val - 1, max_val + 1)

# Save the scatter plot as a JPEG file
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'
plt.savefig(os.path.join(output_dir, 'Daytime_Regression_scatter.jpeg'), format='jpeg')


# Print the regression model summary
print(results.summary())

# Save the regression model summary to a text file
with open(os.path.join(output_dir, 'Daytime_Regression_summary.txt'), 'w') as f:
    f.write(str(results.summary()))

# Create a new DataFrame with the actual and predicted TincO values, and the variables used in the regression model
df_results = df.copy()
df_results['Predicted_TincO'] = predicted_values

# Convert the DataFrame to a string with the values aligned in columns
df_results_str = df_results.to_string()

# Save this string to a text file
with open(os.path.join(output_dir, 'Daytime_Regression_results.txt'), 'w') as f:
    f.write(df_results_str)






#Nighttime regression

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Define the independent variables (x) and the dependent variable (y)
x = df[['S_HGT_AGL', 'LambdaP', 'W_area', 'AVG_Pl_NT']]
y = df['TincI']

# Add a constant column to the independent variables
x = sm.add_constant(x)

# Create the multiple regression model
model = sm.OLS(y, x)

# Fit the model to the data
results = model.fit()

# Get the predicted values
predicted_values = results.predict()

# Create a scatter plot of actual TincO (x-axis) vs. predicted TincO (y-axis)
plt.scatter(y, predicted_values, facecolors='none', edgecolors='blue')
plt.xlabel('Calculated $T_{plan}$ (°C)')
plt.ylabel('Predicted $T_{plan}$ (°C)')
plt.title('Regression Model: Calculated vs. Predicted $T_{plan}$ (°C)')

# Get the minimum and maximum values for x and y
x_min = y.min()
x_max = y.max()
y_min = predicted_values.min()
y_max = predicted_values.max()

# Plot the identity line
min_val = min(x_min, y_min)
max_val = max(x_max, y_max)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values
plt.xlim(min_val - 1, max_val + 1)
plt.ylim(min_val - 1, max_val + 1)

# Save the scatter plot as a JPEG file
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files'
plt.savefig(os.path.join(output_dir, 'Nighttime_Regression_scatter.jpeg'), format='jpeg')


# Print the regression model summary
print(results.summary())

# Save the regression model summary to a text file
with open(os.path.join(output_dir, 'Nighttime_Regression_summary.txt'), 'w') as f:
    f.write(str(results.summary()))

# Create a new DataFrame with the actual and predicted TincO values, and the variables used in the regression model
df_results = df.copy()
df_results['Predicted_TincI'] = predicted_values

# Convert the DataFrame to a string with the values aligned in columns
df_results_str = df_results.to_string()

# Save this string to a text file
with open(os.path.join(output_dir, 'Nighttime_Regression_results.txt'), 'w') as f:
    f.write(df_results_str)


print('DONE CREATING MULTIPLE REGRESSIONS')





















#daytime regression

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Define the independent variables (x) and the dependent variable (y)
x = df[['S_HGT_AGL', 'LambdaP', 'W_area', 'AVG_Pl_DT']]
y = df['TincO']

# Add a constant column to the independent variables
x = sm.add_constant(x)

# Create the multiple regression model
model = sm.OLS(y, x)

# Fit the model to the data
results = model.fit()

# Get the predicted values
predicted_values = results.predict()

# Create a scatter plot of actual TincO (x-axis) vs. predicted TincO (y-axis)
plt.scatter(y, predicted_values, facecolors='none', edgecolors='blue')
plt.xlabel('Actual Tped')
plt.ylabel('Predicted Tped')
plt.title('Regression Model: Actual vs. Predicted Tped')

# Get the minimum and maximum values for x and y
x_min = y.min()
x_max = y.max()
y_min = predicted_values.min()
y_max = predicted_values.max()

# Plot the identity line
min_val = min(x_min, y_min)
max_val = max(x_max, y_max)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values
plt.xlim(min_val - 1, max_val + 1)
plt.ylim(min_val - 1, max_val + 1)

# Save the scatter plot as a JPEG file
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
plt.savefig(os.path.join(output_dir, 'Daytime_Regression_scatter.jpeg'), format='jpeg')


# Print the regression model summary
print(results.summary())

# Save the regression model summary to a text file
with open(os.path.join(output_dir, 'Daytime_Regression_summary.txt'), 'w') as f:
    f.write(str(results.summary()))

# Create a new DataFrame with the actual and predicted TincO values, and the variables used in the regression model
df_results = df.copy()
df_results['Predicted_TincO'] = predicted_values

# Convert the DataFrame to a string with the values aligned in columns
df_results_str = df_results.to_string()

# Save this string to a text file
with open(os.path.join(output_dir, 'Daytime_Regression_results.txt'), 'w') as f:
    f.write(df_results_str)





#Nighttime regression

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Path to the Excel file
file_path = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Final_File_with_coordinates.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Define the independent variables (x) and the dependent variable (y)
x = df[['S_HGT_AGL', 'LambdaP', 'W_area', 'AVG_Pl_NT']]
y = df['TincI']

# Add a constant column to the independent variables
x = sm.add_constant(x)

# Create the multiple regression model
model = sm.OLS(y, x)

# Fit the model to the data
results = model.fit()

# Get the predicted values
predicted_values = results.predict()

# Create a scatter plot of actual TincO (x-axis) vs. predicted TincO (y-axis)
plt.scatter(y, predicted_values, facecolors='none', edgecolors='blue')
plt.xlabel('Actual Tped')
plt.ylabel('Predicted Tped')
plt.title('Regression Model: Actual vs. Predicted Tped')

# Get the minimum and maximum values for x and y
x_min = y.min()
x_max = y.max()
y_min = predicted_values.min()
y_max = predicted_values.max()

# Plot the identity line
min_val = min(x_min, y_min)
max_val = max(x_max, y_max)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Set the axis range based on the minimum and maximum values
plt.xlim(min_val - 1, max_val + 1)
plt.ylim(min_val - 1, max_val + 1)

# Save the scatter plot as a JPEG file
output_dir = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids'
plt.savefig(os.path.join(output_dir, 'Nighttime_Regression_scatter.jpeg'), format='jpeg')


# Print the regression model summary
print(results.summary())

# Save the regression model summary to a text file
with open(os.path.join(output_dir, 'Nighttime_Regression_summary.txt'), 'w') as f:
    f.write(str(results.summary()))

# Create a new DataFrame with the actual and predicted TincO values, and the variables used in the regression model
df_results = df.copy()
df_results['Predicted_TincI'] = predicted_values

# Convert the DataFrame to a string with the values aligned in columns
df_results_str = df_results.to_string()

# Save this string to a text file
with open(os.path.join(output_dir, 'Nighttime_Regression_results.txt'), 'w') as f:
    f.write(df_results_str)
