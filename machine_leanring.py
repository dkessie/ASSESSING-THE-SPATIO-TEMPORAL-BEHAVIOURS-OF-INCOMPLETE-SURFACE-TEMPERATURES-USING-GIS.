#the script below trains but it does not test




import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import numpy as np
import statsmodels.api as sm

def load_data(file_path, sheet_name):
    try:
        xls = pd.ExcelFile(file_path)
        df = pd.read_excel(xls, sheet_name)
        return df
    except Exception as e:
        print("Error in loading data: ", e)
        return None

def create_output_folder(base_output_folder, sheet_name):
    return os.path.join(base_output_folder, sheet_name)

def train_model(df_day, output_folder, sheet_name):
    print("Processing sheet:", sheet_name)
    

    os.makedirs(output_folder, exist_ok=True)  # Create output_folder if not exists

    X_columns = ['LambdaP', 'W_area']
    X_columns.append('AVG_Pl_DT' if 'AVG_Pl_DT' in df_day else 'AVG_Pl_NT')
    if 'Zenith angles' in df_day:
        X_columns.append('Zenith angles')

    y_column = 'TincO' if 'TincO' in df_day else 'TincI'

    df_day = df_day.dropna()  # Drop rows with NaN values
    df_day = df_day[(df_day[X_columns] != 0).all(axis=1)]  # Drop rows with zero values

    X = df_day[X_columns]
    y = df_day[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Add a constant to the independent value
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Perform OLS:
    ols_model = sm.OLS(y_train, X_train)
    model = ols_model.fit()

    predicted_train = model.predict(X_train)

    plt.figure(figsize=(12, 8))
    plt.scatter(y_train, predicted_train, facecolors='none', edgecolors='blue')
    
  

    plt.xlabel('Calculated $T_{ped}$ (°C)', fontsize=18)
    plt.ylabel('Predicted $T_{ped}$ (°C)', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)



    min_value = min(y_train.min(), predicted_train.min())
    max_value = max(y_train.max(), predicted_train.max())
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')

    plt.xlim(min_value - 1, max_value + 1)
    plt.ylim(min_value - 1, max_value + 1)

    plt.savefig(os.path.join(output_folder, 'Scatter_Plot_Train.png'))
    plt.close()

    return model, X_test, y_test

def test_model(model, X_test, y_test, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output_folder if not exists

    predicted_TincO = model.predict(X_test)
    df_test = pd.DataFrame({'TincO': y_test, 'Predicted_TincO': predicted_TincO})
    metrics = calculate_metrics(df_test)

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, predicted_TincO, facecolors='none', edgecolors='blue')
    

    plt.xlabel('Calculated $T_{ped}$ (°C)', fontsize=18)
    plt.ylabel('Predicted $T_{ped}$ (°C)', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)



    min_value = min(y_test.min(), predicted_TincO.min())
    max_value = max(y_test.max(), predicted_TincO.max())
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')

    plt.xlim(min_value - 1, max_value + 1)
    plt.ylim(min_value - 1, max_value + 1)

    plt.savefig(os.path.join(output_folder, 'Scatter_Plot_Test.png'))
    plt.close()

    # Saving the model summary to a text file
    with open(os.path.join(output_folder, 'model_summary.txt'), 'w') as f:
        f.write(model.summary().as_text())
    
    # Saving the metrics to a text file
    with open(os.path.join(output_folder, 'test_metrics.txt'), 'w') as f:
        f.write("RMSE: " + str(metrics[0]) + "\n")
        f.write("R2 Score: " + str(metrics[1]) + "\n")
        f.write("Adjusted R2: " + str(metrics[2]) + "\n")
        f.write("MAPE: " + str(metrics[3]) + "\n")
        f.write("MAE: " + str(metrics[4]) + "\n")

    return metrics

def calculate_metrics(df):
    rmse = sqrt(mean_squared_error(df['TincO'], df['Predicted_TincO']))
    r2 = r2_score(df['TincO'], df['Predicted_TincO'])
    n = df.shape[0]
    p = df.shape[1] - 1
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    mape = np.mean(np.abs((df['TincO'] - df['Predicted_TincO']) / df['TincO'])) * 100
    mae = mean_absolute_error(df['TincO'], df['Predicted_TincO'])
    return (rmse, r2, adjusted_r2, mape, mae)

def main():
    input_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Vancouver Flights.xlsx'
    base_output_folder = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\output'
    sheet_names = ['Flight1_101', 'Flight2_101', 'Flight3_101', 'LCZ5_Day', 
                   'LCZ5_Night', 'LCZ6_Day', 'LCZ6_Night', 'LCZ7_Day',
                    'LCZ7_Night', 'LCZ8_Day', 'LCZ8_Night',
                     'Earth_future_Day', 'Earth_future_Ni', 'Flights_combined', 'ALL_LCZ_100_Day', 'ALL_LCZ_100_Night']
    
    for sheet_name in sheet_names:
        df = load_data(input_file, sheet_name)
        if df is not None:
            output_folder = create_output_folder(base_output_folder, sheet_name)
            model, X_test, y_test = train_model(df, output_folder, sheet_name)  # Add sheet_name parameter
            if model is not None:
                test_metrics = test_model(model, X_test, y_test, output_folder)
                print("Test Metrics for {}: ".format(sheet_name), test_metrics)

if __name__ == "__main__":
    main()

print('DONE WITH MACHINE LEARNING')



#the script below train and test on other areas
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import numpy as np

def load_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        df_day = pd.read_excel(xls, 'Earth_future_Day')
        flight_sheets = ['Flight1_101', 'Flight2_101', 'Flight3_101']
        flight_dfs = [pd.read_excel(xls, sheet) for sheet in flight_sheets]
        return df_day, flight_dfs
    except Exception as e:
        print("Error in loading data: ", e)
        return None, None


def train_model(df_day, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create output_folder if not exists
    X = df_day[['LambdaP', 'W_area', 'AVG_Pl_DT', 'Zenith angles']]
    y = df_day['TincO']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Creating a scatter plot for the trained data
    predicted_train = model.predict(X_train)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, predicted_train)
    min_value = min(y_train.min(), predicted_train.min())
    max_value = max(y_train.max(), predicted_train.max())
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')
    plt.title('Scatter plot for Calculated and Predicted Training Data')
    plt.xlabel('Calculated $T_{ped}$ (°C)')
    plt.ylabel('Predicted $T_{ped}$ (°C)')
    plt.savefig(os.path.join(output_folder, 'Scatter_Plot_Train.png'))
    plt.close()

    return model, X_test, y_test

def test_model(model, X_test, y_test, output_folder):
    predicted_TincO = model.predict(X_test)
    df_test = pd.DataFrame({'TincO': y_test, 'Predicted_TincO': predicted_TincO})
    metrics = calculate_metrics([df_test])

    # Creating a scatter plot for the tested data
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predicted_TincO)
    min_value = min(y_test.min(), predicted_TincO.min())
    max_value = max(y_test.max(), predicted_TincO.max())
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')
    plt.title('Scatter plot for Calculated and Predicted Test Data')
    plt.xlabel('Calculated $T_{ped}$ (°C)')
    plt.ylabel('Predicted $T_{ped}$ (°C)')
    plt.savefig(os.path.join(output_folder, 'Scatter_Plot_Test.png'))
    plt.close()

    return metrics








def predict(flight_dfs, model):
    predicted_dfs = []
    for i, df in enumerate(flight_dfs, start=1):
        X_predict = df[['LambdaP', 'W_area', 'AVG_Pl_DT', 'Zenith angles']]
        predicted_TincO = model.predict(X_predict)
        df['Predicted_TincO'] = predicted_TincO
        df['Flight number'] = 'Flight {}'.format(i)
        predicted_dfs.append(df)
    return predicted_dfs

def calculate_metrics(predicted_dfs):
    metrics = []
    for df in predicted_dfs:
        rmse = sqrt(mean_squared_error(df['TincO'], df['Predicted_TincO']))
        r2 = r2_score(df['TincO'], df['Predicted_TincO'])
        n = df.shape[0]
        p = df.shape[1] - 1
        adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
        mape = np.mean(np.abs((df['TincO'] - df['Predicted_TincO']) / df['TincO'])) * 100
        mae = mean_absolute_error(df['TincO'], df['Predicted_TincO'])
        metrics.append((rmse, r2, adjusted_r2, mape, mae))
    return metrics

def save_metrics(folder_path, metric, flight_number):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'model_metrics.txt'), 'w') as f:
        f.write('Metrics for Flight {}:\n'.format(flight_number))
        f.write('RMSE: {}\n'.format(metric[0]))
        f.write('R^2: {}\n'.format(metric[1]))
        f.write('Adjusted R^2: {}\n'.format(metric[2]))
        f.write('MAPE: {}\n'.format(metric[3]))
        f.write('MAE: {}\n\n'.format(metric[4]))

def save_df(dfs, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    with pd.ExcelWriter(os.path.join(folder_path, 'Vancouver_Flights_modified.xlsx')) as writer:  
        for i, df in enumerate(dfs, start=1):
            df.to_excel(writer, sheet_name='Flight{}_101_modified'.format(i))

def plot_metrics(output_folder, metrics, flights=['Flight 1', 'Flight 2', 'Flight 3']):
    fig, ax = plt.subplots()
    ax.axis('off')

    metric_names = ['Flight', 'RMSE', 'R^2', 'Adjusted R^2', 'MAPE', 'MAE']
    table_data = [metric_names]

    for flight, flight_metrics in zip(flights, metrics):
        flight_data = [flight] + ['{:.2f}'.format(value) for value in flight_metrics]
        table_data.append(flight_data)

    ax.table(cellText=table_data, loc='center', cellLoc='center', rowLoc='center')

    plt.savefig(os.path.join(output_folder, 'metrics_summary.png'))

def create_scatter_plot(predicted_dfs, folder_path):
    for i, df in enumerate(predicted_dfs, start=1):
        plt.figure(figsize=(10, 6))
        plt.scatter(df['TincO'], df['Predicted_TincO'])
        min_value = min(df['TincO'].min(), df['Predicted_TincO'].min())
        max_value = max(df['TincO'].max(), df['Predicted_TincO'].max())
        plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')
        plt.title(f'Scatter plot for Calculated and predicted Tped (°C) for Flight {i}')
        plt.xlabel('Calculated $T_{ped}$ (°C)')
        plt.ylabel('Predicted $T_{ped}$ (°C)')
        plt.savefig(os.path.join(folder_path, f'Scatter_Plot_Flight{i}.png'))
        plt.close()

def save_metrics(folder_path, metric, flight_number):
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, 'model_metrics.txt'), 'w') as f:
        f.write('Metrics for Flight {}:\n'.format(flight_number))
        f.write('RMSE: {}\n'.format(metric[0]))
        f.write('R^2: {}\n'.format(metric[1]))
        f.write('Adjusted R^2: {}\n'.format(metric[2]))
        f.write('MAPE: {}\n'.format(metric[3]))
        f.write('MAE: {}\n\n'.format(metric[4]))


def main():
    input_file = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\Vancouver Flights.xlsx'
    output_folder = r'C:\Users\kwame\OneDrive - The University of Western Ontario\Desktop\GOOGLE DRIVE\NEW CALCULATION\grids\Needed files\output'
    df_day, flight_dfs = load_data(input_file)
    model, X_test, y_test = train_model(df_day, output_folder)  # Included output_folder
    test_metrics = test_model(model, X_test, y_test, output_folder)  # Included output_folder
    print("Test Metrics: ", test_metrics)
    save_metrics(output_folder, test_metrics[0], flight_number='Test')
    predicted_dfs = predict(flight_dfs, model)
    metrics = calculate_metrics(predicted_dfs)
    for i, _ in enumerate(flight_dfs, start=1):
        save_metrics(os.path.join(output_folder, 'Flight{}_101'.format(i)), metrics[i-1], i)
        save_df([predicted_dfs[i-1]], os.path.join(output_folder, 'Flight{}_101'.format(i)))
    create_scatter_plot(predicted_dfs, output_folder)
    plot_metrics(output_folder, metrics)

if __name__ == "__main__":
    main()

print('DONE WITH MACHINE LEARNING')
